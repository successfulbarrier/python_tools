#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:ultralytics_yolov3
# author:机灵巢穴_WitNest
# datetime:2023/9/8 16:08
# software: PyCharm
# -*- coding: utf-8 -*-
"""
    yolo数据集可视化
"""
import cv2
import os
import numpy as np
from pathlib import Path

id2cls = {0: 'pig'}
cls2id = {'pig': 0}
id2color = {0: (0, 255, 0)}


# 支持中文路径
def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    return cv_img


def vis_yolo(yolo_image_dir, yolo_label_dir, save_dir='res/'):
    yolo_image_dir = str(Path(yolo_image_dir)) + '/'
    yolo_label_dir = str(Path(yolo_label_dir)) + '/'
    save_dir = str(Path(save_dir)) + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_files = os.listdir(yolo_image_dir)
    for iimgf, imgf in enumerate(image_files):
        print(iimgf + 1, '/', len(image_files), imgf)
        fn = imgf.rsplit('.', 1)[0]
        image = cv_imread(yolo_image_dir + imgf)
        h, w = image.shape[:2]
        if not os.path.exists(yolo_label_dir + fn + '.txt'):
            continue
        labels = np.loadtxt(yolo_label_dir + fn + '.txt').reshape(-1, 5)
        if len(labels) > 0:
            labels[:, 1::2] = w * labels[:, 1::2]
            labels[:, 2::2] = h * labels[:, 2::2]
            labels_xyxy = np.zeros(labels.shape)
            labels_xyxy[:, 1] = np.clip(labels[:, 1] - labels[:, 3] / 2, 0, w)
            labels_xyxy[:, 2] = np.clip(labels[:, 2] - labels[:, 4] / 2, 0, h)
            labels_xyxy[:, 3] = np.clip(labels[:, 1] + labels[:, 3] / 2, 0, w)
            labels_xyxy[:, 4] = np.clip(labels[:, 2] + labels[:, 4] / 2, 0, h)
            for label in labels_xyxy:
                color = id2color[int(label[0])]
                x1, y1, x2, y2 = label[1:]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.imencode(os.path.splitext(imgf)[-1], image)[1].tofile(save_dir + imgf)
    print('Completed!')


if __name__ == '__main__':
    yolo_image_dir = r'H:\Data\pigs\images\train'
    yolo_label_dir = r'H:\Data\pigs\labels\train'
    save_dir = r'res1/'
    vis_yolo(yolo_image_dir, yolo_label_dir, save_dir)



