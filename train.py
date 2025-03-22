# %%
from ultralytics import YOLO


# %%
model = YOLO("yolo11n.pt")

# %%
train_results = model.train(
    data="/run/media/aun1x/New Volume1/final year project/Weeds.v3-augmented_nottrained.yolov8/data.yaml",  # path to dataset YAML
    epochs=50,  # number of training epochs
    imgsz=640,  # training image size
    amp=False,
    batch=8,
    name='take1_50e_8b_y11'
)


# %%



