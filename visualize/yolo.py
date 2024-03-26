# visualize/yolo.py
# This file contains the code to visualize the output of the YOLO model.

import cv2
import numpy as np
import ultralytics


def visualize_yolo(image, model, class_names, conf_threshold=0.01, iou_threshold=0.3):
    model.predict(image, conf=conf_threshold, iou=iou_threshold, save=True)
