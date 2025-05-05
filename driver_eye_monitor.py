import time
import torch
import cv2
import messaging
from ultralytics import YOLO
from cereal import log

# Constants
MODEL_PATH = "yolov8n_eye.pt"
EYE_CLASSES = {0: "Open Eye", 1: "Closed Eye"}
CLOSED_CLASS_ID = 1
OPEN_CLASS_ID = 0
CLOSED_THRESHOLD_SEC = 5
EMERGENCY_GRACE_SEC = 3
INFERENCE_INTERVAL = 0.5  # seconds between frames

# Messaging
pm = messaging.PubMaster(['driverMonitoringState', 'controlsState'])

# Model
model = YOLO(MODEL_PATH)
model.fuse()

# State
closed_start_time = None
alarm_triggered = False

def send_drowsiness_alert():
    print("⚠️ Sending DROWSINESS alert to driverMonitoringState...")
    dm_msg = log.Event.new_message("driverMonitoringState")
    dm_msg.driverMonitoringState.faceDetected = True
    dm_msg.driverMonitoringState.isDistracted = True
    dm_msg.driverMonitoringState.alertType = log.DriverMonitoringState.AlertType.DROWSY
    pm.send("driverMonitoringState", dm_msg)

def send_emergency_stop():
    print("🛑 Sending EMERGENCY STOP to controlsState...")
    cs_msg = log.Event.new_message("controlsState")
    cs_msg.controlsState.enabled = False
    cs_msg.controlsState.alert = "EMERGENCY_STOP"
    cs_msg.controlsState.forceDecel = True
    pm.send("controlsState", cs_msg)

def monitor_frame(frame):
    global closed_start_time, alarm_triggered

    results = model.predict(frame, conf=0.5, classes=[0, 1], verbose=False)[0]
    class_ids = [int(box.cls) for box in results.boxes]

    eye_closed = CLOSED_CLASS_ID in class_ids and OPEN_CLASS_ID not in class_ids

    if eye_closed:
        if closed_start_time is None:
            closed_start_time = time.monotonic()
        else:
            duration = time.monotonic() - closed_start_time
            if duration > CLOSED_THRESHOLD_SEC and not alarm_triggered:
                send_drowsiness_alert()
                alarm_triggered = True
            if duration > CLOSED_THRESHOLD_SEC + EMERGENCY_GRACE_SEC:
                send_emergency_stop()
    else:
        closed_start_time = None
        alarm_triggered = False

def main():
    cap = cv2.VideoCapture(0)  # Update to match your driver-facing camera ID
    if not cap.isOpened():
        print("❌ Error: Could not open camera.")
        return

    print("📷 Driver eye monitoring started...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            monitor_frame(frame)
            time.sleep(INFERENCE_INTERVAL)
    except KeyboardInterrupt:
        print("🛑 Interrupted by user.")
    finally:
        cap.release()

if __name__ == "__main__":
    main()
