import yaml
import ultralytics
from ultralytics import YOLO
import os

# current directory base path
os.chdir(os.path.dirname(__file__))

# Load the segmentation model
model = YOLO("yolov8n.pt")  # Load a pretrained model (recommended for training)

# Train the model using the temporary YAML file
results = model.train(
    data="./data.yaml",
    single_cls=True,
    imgsz=512,
    epochs=10,
    device="mps",
    batch=1,
    save_period=5,
    augment=False,  # Enable built-in augmentations
    mosaic=1.0,    # Enable mosaic augmentation
    degrees=20.0,  # Random rotation
    scale=0.9,     # Random scaling
    hsv_h=0.015,   # HSV augmentation
    hsv_s=0.7,
    hsv_v=0.4,
)
