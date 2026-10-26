
import cv2
import time
import threading
import numpy as np
import pygame
from ultralytics import YOLO

# ---------------- NOKKUKUTHI SETTINGS ----------------

MODEL_NAME = "yolov8n.pt"
CAMERA_INDEX = 0
CONFIDENCE = 0.40
ALARM_COOLDOWN = 0.35

# COCO classes supported by this pretrained model
TARGETS = {
    "person", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe","pig","monkey"
}
ALARM_TARGETS = TARGETS - {"person"}


# ------------------------------------------------------

print("Starting NOKKUKUTHI...")
print("Loading AI model. The first run may take a while.")

model = YOLO(MODEL_NAME)

# Set up a safe, audible computer-generated alarm
pygame.mixer.init()
sample_rate = 22050
duration = 10.0
t = np.arange(int(sample_rate * duration)) / sample_rate
wave = (10000 * np.sin(2 * np.pi * 880 * t)).astype(np.int16)
stereo_wave = np.column_stack((wave, wave))
alarm_sound = pygame.sndarray.make_sound(stereo_wave)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(
        "Camera not found. Close other camera apps "
        "or change CAMERA_INDEX."
    )

zone = None
drag_start = None
dragging = False
previous_inside = False
last_alarm = 0
alarm_lock = threading.Lock()
alarm_playing = False


def play_alarm():
    """Play a short beep without blocking camera processing."""
    global alarm_playing
    with alarm_lock:
        if alarm_playing:
            return
        alarm_playing = True

    def worker():
        global alarm_playing
        try:
            alarm_sound.play()
            time.sleep(duration + 0.1)
        finally:
            with alarm_lock:
                alarm_playing = False

    threading.Thread(target=worker, daemon=True).start()


def mouse_callback(event, x, y, flags, param):
    """Drag on the video to choose the protected crop zone."""
    global zone, drag_start, dragging

    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = (x, y)
        dragging = True

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        zone = (
            min(drag_start[0], x),
            min(drag_start[1], y),
            max(drag_start[0], x),
            max(drag_start[1], y)
        )

    elif event == cv2.EVENT_LBUTTONUP and dragging:
        zone = (
            min(drag_start[0], x),
            min(drag_start[1], y),
            max(drag_start[0], x),
            max(drag_start[1], y)
        )
        dragging = False

        if zone[2] - zone[0] < 20 or zone[3] - zone[1] < 20:
            zone = None


window_name = "NOKKUKUTHI - Smart Crop Protection"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

print("Camera started.")
print("Drag to mark the crop zone | R: reset zone | Q: quit")

try:
    while True:
        ok, frame = cap.read()

        if not ok:
            print("Could not read camera frame.")
            break

        height, width = frame.shape[:2]

        # Default protected area: central part of the view
        if zone is None and not dragging:
            active_zone = (
                int(width * 0.20),
                int(height * 0.20),
                int(width * 0.80),
                int(height * 0.80)
            )
        else:
            active_zone = zone

        # Run AI detection
        results = model.predict(
            frame,
            conf=CONFIDENCE,
            imgsz=640,
            verbose=False
        )

        inside_now = False
        detected_names = []

        if active_zone is not None:
            x1, y1, x2, y2 = active_zone
            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                (0, 255, 255), 2
            )
            cv2.putText(
                frame, "PROTECTED CROP ZONE",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2
            )

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                name = model.names[cls_id]

                if name not in TARGETS:
                    continue

                confidence = float(box.conf[0])
                bx1, by1, bx2, by2 = map(
                    int, box.xyxy[0].tolist()
                )

                # Use the bottom-center of the bounding box
                # as the object's approximate ground position.
                foot_x = (bx1 + bx2) // 2
                foot_y = by2

                in_zone = False
                if active_zone is not None:
                    zx1, zy1, zx2, zy2 = active_zone
                    in_zone = (
                        zx1 <= foot_x <= zx2
                        and zy1 <= foot_y <= zy2
                    )

                                # Red only for target animals inside the protected zone
                if in_zone and name != "person":
                    color = (0, 0, 255)  # Red
                else:
                    color = (0, 255, 0)  # Green

                cv2.rectangle(
                    frame, (bx1, by1), (bx2, by2), color, 2
                )

                label = f"{name}: {confidence:.0%}"
                cv2.putText(
                    frame, label,
                    (bx1, max(20, by1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 2
                )

            if in_zone and name in ALARM_TARGETS:
                inside_now = True
                detected_names.append(name)

        # Trigger only when the zone changes from empty to occupied.
        current_time = time.time()

        if (
            inside_now
            and not previous_inside
            and current_time - last_alarm >= ALARM_COOLDOWN
        ):
            play_alarm()
            last_alarm = current_time

        previous_inside = inside_now

        if inside_now:
            status = "ANIMAL DETECTED!"
            status_color = (0, 0, 255)  # Red
        else:
            status = "AREA SAFE - MONITORING"
            status_color = (0, 200, 0)  # Green

        cv2.putText(
            frame, status, (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            status_color, 2
        )

        cv2.putText(
            frame,
            "Drag: set zone | R: reset | Q: quit",
            (15, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 2
        )

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            zone = None
            dragging = False
            drag_start = None

finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    print("NOKKUKUTHI stopped.")
