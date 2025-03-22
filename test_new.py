import cv2
from ultralytics import YOLO

# Load a YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("/run/media/aun1x/New Volume/final year project/runs/detect/100e_b8_y83/weights/best.pt") 

# result = model("/run/media/aun1x/New Volume1/final year project/Weeds.v3-augmented_nottrained.yolov8/test/images/20210907_153931_x264_mp4-184_jpg.rf.d84795ffcfda403b97a022567dc0cde6.jpg")
# result = model("/run/media/aun1x/New Volume1/final year project/Weeds.v3-augmented_nottrained.yolov8/test/images/20210907_153931_x264_mp4-170_jpg.rf.75e8d2e5f6cf08dab4996f6e1eda9704.jpg")
result = model("/run/media/aun1x/New Volume/final year project/Weeds.v3-augmented_nottrained.yolov8/test/images/20210907_153931_x264_mp4-72_jpg.rf.dc205a989ba2a8467f257dc3ae3a46dd.jpg")
result[0].show()

