# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   data_set_concat_class.py
# @Time    :   2024/07/21 15:57:05
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   将yolo格式的标签进行类别合并

import os
import shutil
import subprocess

#-------------------------------------------------#
#   文件路径
#-------------------------------------------------#
yolo_lable_path = "/media/lht/D_Project/datasets/light_dataset2/train/labels"
new_yolo_lable_path = "/media/lht/D_Project/datasets/light_dataset2/train/labels_new"

#-------------------------------------------------#
#   要合并的类别，合并后的类别序号默认采用合并类别中最小的类别序号
#-------------------------------------------------#
concat_class = [0,1,2,3,4,5,6,7]

#-------------------------------------------------#
#   创建新标签存放路径
#-------------------------------------------------#
if not os.path.exists(new_yolo_lable_path):
    os.makedirs(new_yolo_lable_path)

#-------------------------------------------------#
#   合并类别
#-------------------------------------------------#
def merge_classes(label, concat_class):
    if label in concat_class:
        return min(concat_class)
    else:
        if label == 9:
            if label > min(concat_class):
                return label - (len(concat_class)-1)
            else:
                return label
        else:
            if label > min(concat_class):
                return label - (len(concat_class)-1)
            else:
                return label
#-------------------------------------------------#
#   处理标签文件
#-------------------------------------------------#
label_files = os.listdir(yolo_lable_path)
for file in label_files:
    with open(os.path.join(yolo_lable_path, file), 'r') as f:
        lines = f.readlines()
    
    with open(os.path.join(new_yolo_lable_path, file), 'w') as f:
        for line in lines:
            label = int(line.split()[0])
            new_label = merge_classes(label, concat_class)
            f.write(f"{int(new_label)} {' '.join(line.split()[1:])}"+"\n")

print("标签合并完成！")

#-------------------------------------------------#
#   获取类别信息
#-------------------------------------------------#
subprocess.run(["python", "data_set/data_set_read_class.py", 
                "--yolo_lable_path", new_yolo_lable_path])