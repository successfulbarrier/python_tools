# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   data_set_extract_new_dataset.py
# @Time    :   2024/07/21 16:24:19
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   从一个类别较多的yolo格式的目标检测数据集中提取出指定的几个类别的数据集
import os
import cv2
import random
import string
import shutil
from tqdm import tqdm

#-------------------------------------------------#
#   检测数据集的路径数
#-------------------------------------------------#
Detection_root = "/media/lht/LHT/code/datasets/light_dataset2_"
Detection_train_images = Detection_root + "/train/images"
Detection_train_labels = Detection_root + "/train/labels"
Detection_val_images = Detection_root + "/val/images"
Detection_val_labels = Detection_root + "/val/labels"

#-------------------------------------------------#
#   提取出来的新的检测数据集
#-------------------------------------------------#
new_dataset_root = Detection_root + "/new_dataset3"
new_dataset_train_images = new_dataset_root + "/train/images"
new_dataset_train_labels = new_dataset_root + "/train/labels"
new_dataset_val_images = new_dataset_root + "/val/images"
new_dataset_val_labels = new_dataset_root + "/val/labels"

#-------------------------------------------------#
#   要提取的类别
#-------------------------------------------------#
# class_name  = ["light_off_red", "light_on_red", "light_off_green", "light_on_green", 
#                "light_off_yellow", "light_on_yellow", "light_off_white", "light_on_white", "group_red", "group_green"]
# class_id    = [0, 1, 2, 3, 4, 5, 6, 7, 20, 21]

# class_name  = ["switch_one_0", "switch_one_270", "switch_two_0", "switch_two_270", 
#                "switch_three_0", "switch_three_270", "switch_four_0", "switch_four_270",
#                "switch_five_0", "switch_five_270", "switch_five_90"]
# class_id    = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22]

class_name  = ["ya_ban_off", "ya_ban_on"]
class_id    = [18, 19]

# 生成一个包含所有字母和数字的列表
all_chars = string.ascii_letters + string.digits

# 创建分类数据集存放路径
if not os.path.exists(new_dataset_root):
    os.makedirs(new_dataset_root)
if not os.path.exists(new_dataset_train_images):
    os.makedirs(new_dataset_train_images)
if not os.path.exists(new_dataset_train_labels):
    os.makedirs(new_dataset_train_labels)
if not os.path.exists(new_dataset_val_images):
    os.makedirs(new_dataset_val_images)
if not os.path.exists(new_dataset_val_labels):
    os.makedirs(new_dataset_val_labels)
    
# 提取类别并保存到分类数据集
def extract_class_images(input_image_path, input_label_path, output_image_path, output_label_path, class_id, class_name):
    label_files = os.listdir(input_label_path)
    for file in tqdm(label_files):
        with open(os.path.join(input_label_path, file), 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            label = int(line.split()[0])
            if label in class_id:
                class_index = class_id.index(label)
                new_lines.append(f"{class_index} {' '.join(line.split()[1:])}\n")

        if len(new_lines) > 0:    
            with open(os.path.join(output_label_path, file), 'w') as f:
                f.writelines(new_lines)
            image_file = os.path.join(input_image_path, file.replace(".txt", ".jpg"))
            if not os.path.exists(image_file):
                image_file = os.path.join(input_image_path, file.replace(".txt", ".png"))
            shutil.copy(image_file, output_image_path)

                        
# 提取训练集类别图片
extract_class_images(Detection_train_images, Detection_train_labels, new_dataset_train_images, new_dataset_train_labels, class_id, class_name)

# 提取验证集类别图片
extract_class_images(Detection_val_images, Detection_val_labels, new_dataset_val_images, new_dataset_val_labels, class_id, class_name)

print("分类数据集提取完成！")

