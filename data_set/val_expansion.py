# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   val_expansion.py
# @Time    :   2024/10/12 15:44:26
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   扩充验证集数据到指定数量

import os
import random
import shutil

#-------------------------------------------------#
#   检测数据集的路径数
#-------------------------------------------------#
Detection_root = "/media/lht/LHT/code/datasets/TS_dataset5"
Detection_train_images = Detection_root + "/train/images"
Detection_train_labels = Detection_root + "/train/labels"
Detection_val_images = Detection_root + "/val/images"
Detection_val_labels = Detection_root + "/val/labels"

#-------------------------------------------------#
#   验证集要扩充到的数量
#-------------------------------------------------#
val_num = 1143


#-------------------------------------------------#
#   获取现有训练集和验证集的数量，以及文件列表
#-------------------------------------------------#
train_image_files = os.listdir(Detection_train_images)
val_image_files = os.listdir(Detection_val_images)
train_num_now = len(train_image_files)
val_num_now = len(val_image_files)

val_differ_num = val_num-val_num_now
if val_differ_num <=0:
    raise "不需要添加！！！"


#-------------------------------------------------#
#   从训练集中随机提取图片和标签加入到验证集
#-------------------------------------------------#
while val_num_now <= val_num:
    random_index = random.randint(0, train_num_now-1)
    image_file = train_image_files[random_index]
    label_file = os.path.splitext(image_file)[0] + ".txt"
    
    shutil.copy(os.path.join(Detection_train_images, image_file), Detection_val_images)
    shutil.copy(os.path.join(Detection_train_labels, label_file), Detection_val_labels)
    
    val_num_now += 1

print("制作完毕！！")







