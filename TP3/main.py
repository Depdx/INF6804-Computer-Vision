import cv2
from ultralytics import YOLO

# Load an official or custom model
model = YOLO("yolov8x.pt")  # Load an official Detect model

# Perform tracking with the model
results = model.track(
    source="data/video.mp4",
    show=False,
    save=True,
    name="test_result",
    persist=True,
    classes=41,
)  # Tracking with default tracker
print(results)
