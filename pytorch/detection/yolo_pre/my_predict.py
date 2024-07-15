#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:ultralytics_yolov8-embeddings
# author:机灵巢穴_WitNest
# datetime:2023/10/20 16:05
# software: PyCharm

# 这里直接调用包实现的预测
from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    model = YOLO("best.pt")
    im2 = cv2.imread("light.jpg")
    results = model.predict(source=im2, save=True, save_txt=True)

