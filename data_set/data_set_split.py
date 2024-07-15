#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:ultralytics_yolov3
# author:机灵巢穴_WitNest
# datetime:2023/9/22 21:11
# software: PyCharm

"""
    将数据集划分成验证集和训练集
"""

import shutil
import os
import random
from tqdm import tqdm


image_path = "D:\\Dataset_light\\new\\images"  # 图片路径
label_path = "D:\\Dataset_light\\new\\labels"  # 标注文件路径
img_tail = ".jpg"   # 图片后缀
label_tail = ".txt"     # 标签文件后缀
dataset_save_path = "D:\\Dataset_light\\new"  # 分好的数据集保存位置
val_num = 10    # 验证集的数量


# 读取文件列表
images = os.listdir(image_path)
random.shuffle(images)

# print(image_path + "\\" + images[0])

# 自动创建目录
os.makedirs(dataset_save_path+"\\train\\images\\")
os.makedirs(dataset_save_path+"\\train\\labels\\")
os.makedirs(dataset_save_path+"\\val\\images\\")
os.makedirs(dataset_save_path+"\\val\\labels\\")

# 拷贝文件
i = 0
file_len = len(images)
train_len = file_len - val_num

for img in tqdm(images):
    if i < train_len:
        shutil.copyfile(image_path+"\\"+img, dataset_save_path+"\\train\\images\\"+img)
        shutil.copyfile(label_path+"\\"+img[:-len(img_tail)]+label_tail, dataset_save_path+"\\train\\labels\\"+img[:-len(img_tail)]+label_tail)
    else:
        shutil.copyfile(image_path+"\\"+img, dataset_save_path+"\\val\\images\\"+img)
        shutil.copyfile(label_path+"\\"+img[:-len(img_tail)]+label_tail, dataset_save_path+"\\val\\labels\\"+img[:-len(img_tail)]+label_tail)
    i += 1

