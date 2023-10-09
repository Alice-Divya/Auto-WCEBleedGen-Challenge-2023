#Importing the ultralytics
import ultralytics
ultralytics.checks()

import numpy as np
# Importing the YOLO from ultralytics
from ultralytics import YOLO
# Loading the yolov8-seg.pt pretarined weights
model = YOLO('yolov8n-seg.pt')
# Training the model with dataset.yaml 
# dataset.yaml contains the paths for train and test
model.train(data="/home/teai/wcebleed/yolodata/dataset.yaml", epochs=250, imgsz=640,seed=3,patience=0,batch=16,name="segmentation")
