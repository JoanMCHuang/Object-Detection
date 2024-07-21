from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'

# Load a model
model = YOLO('yolov8n.pt')  # load an official detection model
model = YOLO('yolov8n-seg.pt')  # load an official segmentation model

# Video Tracking
results = model.track(source="Hsinchu SP.mp4", show=True) 