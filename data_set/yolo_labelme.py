#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:ultralytics_yolov3
# author:机灵巢穴_WitNest
# datetime:2023/9/8 16:05
# software: PyCharm
# -*- coding: utf-8 -*-
"""
    yolo数据集转labelme标注数据集
"""

import cv2
import os
import json
import shutil
import numpy as np
from pathlib import Path

id2cls = {0: 'pig'}


def xyxy2labelme(labels, w, h, image_path, save_dir='res/'):
    save_dir = str(Path(save_dir)) + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    label_dict = {}
    label_dict['version'] = '5.0.1'
    label_dict['flags'] = {}
    label_dict['imageData'] = None
    label_dict['imagePath'] = image_path
    label_dict['imageHeight'] = h
    label_dict['imageWidth'] = w
    label_dict['shapes'] = []
    for l in labels:
        tmp = {}
        tmp['label'] = id2cls[int(l[0])]
        tmp['points'] = [[l[1], l[2]], [l[3], l[4]]]
        tmp['group_id'] = None
        tmp['shape_type'] = 'rectangle'
        tmp['flags'] = {}
        label_dict['shapes'].append(tmp)
    fn = save_dir + image_path.rsplit('.', 1)[0] + '.json'
    with open(fn, 'w') as f:
        json.dump(label_dict, f)


def yolo2labelme(yolo_image_dir, yolo_label_dir, save_dir='res/'):
    yolo_image_dir = str(Path(yolo_image_dir)) + '/'
    yolo_label_dir = str(Path(yolo_label_dir)) + '/'
    save_dir = str(Path(save_dir)) + '/'
    image_files = os.listdir(yolo_image_dir)
    for iimgf, imgf in enumerate(image_files):
        print(iimgf + 1, '/', len(image_files), imgf)
        fn = imgf.rsplit('.', 1)[0]
        shutil.copy(yolo_image_dir + imgf, save_dir + imgf)
        image = cv2.imread(yolo_image_dir + imgf)
        h, w = image.shape[:2]
        if not os.path.exists(yolo_label_dir + fn + '.txt'):
            continue
        labels = np.loadtxt(yolo_label_dir + fn + '.txt').reshape(-1, 5)
        if len(labels) < 1:
            continue
        labels[:, 1::2] = w * labels[:, 1::2]
        labels[:, 2::2] = h * labels[:, 2::2]
        labels_xyxy = np.zeros(labels.shape)
        labels_xyxy[:, 1] = np.clip(labels[:, 1] - labels[:, 3] / 2, 0, w)
        labels_xyxy[:, 2] = np.clip(labels[:, 2] - labels[:, 4] / 2, 0, h)
        labels_xyxy[:, 3] = np.clip(labels[:, 1] + labels[:, 3] / 2, 0, w)
        labels_xyxy[:, 4] = np.clip(labels[:, 2] + labels[:, 4] / 2, 0, h)
        xyxy2labelme(labels_xyxy, w, h, imgf, save_dir)
    print('Completed!')


if __name__ == '__main__':
    yolo_image_dir = r'H:\Data\pigs\images\train'
    yolo_label_dir = r'H:\Data\pigs\labels\train'
    save_dir = r'res/'
    yolo2labelme(yolo_image_dir, yolo_label_dir, save_dir)