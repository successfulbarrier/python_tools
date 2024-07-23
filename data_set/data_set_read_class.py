# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   data_set_read_class.py
# @Time    :   2024/07/21 15:37:25
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   获取yolo格式数据集的所有类别信息，一般用于验证类别信息是否正确

import os
import argparse

#-------------------------------------------------#
#   获取参数
#-------------------------------------------------#
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--yolo_lable_path', type=str, 
                    default="/media/lht/D_Project/datasets/light_dataset2/val/labels", 
                    help='name of the user')
args = parser.parse_args()

#-------------------------------------------------#
#   标签文件路径
#-------------------------------------------------#
# yolo_lable_path = "/media/lht/LHT/code/datasets/TS_dataset5/train/labels"
# yolo_lable_path = "/media/lht/LHT/code/datasets/light_dataset/train/labels"
yolo_lable_path = args.yolo_lable_path

# 读取所有标签文件
label_files = os.listdir(yolo_lable_path)

# 统计不同类别的标签并按顺序排列
label_list = []
for file in label_files:
    with open(os.path.join(yolo_lable_path, file), 'r') as f:
        for line in f:
            label = line.split()[0]
            label_list.append(label)

# 去重并排序
label_set = sorted(set(map(int, label_list)))

# 打印排序后的标签
print(f"总共有{len(label_set)}种标签：{label_set}")
