# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   data_set_extract_classification.py
# @Time    :   2024/07/21 16:24:19
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   从yolo格式的目标检测数据集中提取某些类别的分类数据集
import os
import cv2
import random
import string

#-------------------------------------------------#
#   检测数据集的路径数
#-------------------------------------------------#
Detection_root = "/media/lht/LHT/code/datasets/light_dataset2"
Detection_train_images = Detection_root + "/train/images"
Detection_train_labels = Detection_root + "/train/labels"
Detection_val_images = Detection_root + "/val/images"
Detection_val_labels = Detection_root + "/val/labels"

#-------------------------------------------------#
#   分类数据集的路径数
#-------------------------------------------------#
Classification_root = Detection_root + "/Classification"
Classification_train = Classification_root + "/train"
Classification_val = Classification_root + "/val"

#-------------------------------------------------#
#   要提取的类别
#-------------------------------------------------#
class_name  = ["light_off_red", "light_on_red", "light_off_green", "light_on_green", 
               "light_off_yellow", "light_on_yellow", "light_off_white", "light_on_white"]
class_id    = [0, 1, 2, 3, 4, 5, 6, 7]

# 生成一个包含所有字母和数字的列表
all_chars = string.ascii_letters + string.digits

# 创建分类数据集存放路径
if not os.path.exists(Classification_train):
    os.makedirs(Classification_train)
if not os.path.exists(Classification_val):
    os.makedirs(Classification_val)

# 提取类别并保存到分类数据集
def extract_class_images(input_image_path, input_label_path, output_root, class_id, class_name):
    for i, (id, name) in enumerate(zip(class_id, class_name)):
        class_folder_train = os.path.join(Classification_train, name)
        class_folder_val = os.path.join(Classification_val, name)
        if not os.path.exists(class_folder_train):
            os.makedirs(class_folder_train)
        if not os.path.exists(class_folder_val):
            os.makedirs(class_folder_val)
        
        label_files = os.listdir(input_label_path)
        for file in label_files:
            with open(os.path.join(input_label_path, file), 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                label = int(line.split()[0])
                if label == id:
                    image_file = os.path.join(input_image_path, file.replace(".txt", ".jpg"))
                    image = cv2.imread(image_file)
                    box = [float(x) for x in line.split()[1:]]  # 获取box坐标信息
                    x, y, w, h = box[0] * image.shape[1], box[1] * image.shape[0], box[2] * image.shape[1], box[3] * image.shape[0]  # 归一化坐标乘以图片宽高
                    cropped_image = image[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]  # 截取box区域
                    if cropped_image is not None and cropped_image.size != 0:
                        # 生成长度为10的随机字符串
                        random_string = ''.join(random.sample(all_chars, 10))
                        cv2.imwrite(os.path.join(output_root, name, random_string+".jpg"), cropped_image)  # 保存截取的区域
                    else:
                        print("次文件出现异常："+file)
# 提取训练集类别图片
extract_class_images(Detection_train_images, Detection_train_labels, Classification_train, class_id, class_name)

# 提取验证集类别图片
extract_class_images(Detection_val_images, Detection_val_labels, Classification_val, class_id, class_name)

print("分类数据集提取完成！")

