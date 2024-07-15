#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:ultralytics_yolov3
# author:机灵巢穴_WitNest
# datetime:2023/9/8 16:07
# software: PyCharm
# -*- coding: utf-8 -*-
"""
    labelme数据集可视化
"""
import cv2
import os
import json
import shutil
import numpy as np
from pathlib import Path
from glob import glob

id2cls = {0: 'pig'}
cls2id = {'pig': 0}
id2color = {0: (0, 255, 0)}


# 支持中文路径
def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    return cv_img


def get_labelme_info(label_file):
    anno = json.load(open(label_file, "r", encoding="utf-8"))
    shapes = anno['shapes']
    image_path = os.path.basename(anno['imagePath'])
    labels = []
    for s in shapes:
        pts = s['points']
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        color = id2color[cls2id[s['label']]]
        labels.append([color, x1, y1, x2, y2])
    return labels, image_path


def vis_labelme(labelme_label_dir, save_dir='res/'):
    labelme_label_dir = str(Path(labelme_label_dir)) + '/'
    save_dir = str(Path(save_dir)) + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    json_files = glob(labelme_label_dir + '*.json')
    for ijf, jf in enumerate(json_files):
        print(ijf + 1, '/', len(json_files), jf)
        filename = os.path.basename(jf).rsplit('.', 1)[0]
        labels, image_path = get_labelme_info(jf)
        image = cv_imread(labelme_label_dir + image_path)
        for label in labels:
            color = label[0]
            x1, y1, x2, y2 = label[1:]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        # 显示图片
        # cv2.imshow(filename, image)
        # cv2.waitKey(0)
        # 支持中文路径，保存图片
        cv2.imencode(os.path.splitext(image_path)[-1], image)[1].tofile(save_dir + image_path)
    print('Completed!')


if __name__ == '__main__':
    root_dir = r'D:\tmp\images'
    save_dir = r'res'
    vis_labelme(root_dir, save_dir)