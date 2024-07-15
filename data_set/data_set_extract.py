#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:python_tools
# author:机灵巢穴_WitNest
# datetime:2023/10/28 19:38
# software: PyCharm

"""
    在数据集标注过程中会存在有些照片不可用，因此标注完成之后需要抽出有标签的图片和对应的标签
    本脚本的功能就是将数据集中标注和为标注的图片分开
"""

import shutil
import os
from tqdm import tqdm
import argparse

# 创建ArgumentParser对象
parser = argparse.ArgumentParser(description='Process some integers.')

# 传递带值的参数
parser.add_argument('--dataset_root', type=str, 
                    default="H:\\LHT\\数据集原始数据\\light_dataset\\new_add_data", help='name of the user')

# 解析参数
args = parser.parse_args()

# dataset_root = "H:\\LHT\\数据集原始数据\\light_dataset\\new_add_data"

image_path = args.dataset_root + "\\images"  # 图片路径
label_path = args.dataset_root + "\\jsons"  # 标注文件路径
img_tail = [".jpg", ".JPG"]   # 图片后缀
label_tail = ".json"     # 标签文件后缀
dataset_save_path = args.dataset_root  # 分好的数据集保存位置

# 读取文件列表
images = os.listdir(image_path)

dataset_save_path = dataset_save_path + "\\new_dataset\\"
T_images_path = dataset_save_path + "T_images\\"
T_labrls_path = dataset_save_path + "T_labels\\"
F_images_path = dataset_save_path + "F_images\\"

# 创建保存目录
if not os.path.exists(T_images_path):
    os.makedirs(T_images_path)
if not os.path.exists(T_labrls_path):
    os.makedirs(T_labrls_path)
if not os.path.exists(F_images_path):
    os.makedirs(F_images_path)

# 遍历每张图片
for img in tqdm(images):
    if os.path.exists(label_path+"\\"+img[:-len(img_tail[0])]+label_tail):
        shutil.copyfile(image_path + "\\" + img, T_images_path + img)
        shutil.copyfile(label_path + "\\" + img[:-len(img_tail[0])] + label_tail,
                        T_labrls_path + img[:-len(img_tail[0])] + label_tail)
    elif os.path.exists(label_path+"\\"+img[:-len(img_tail[1])]+label_tail):
        shutil.copyfile(image_path + "\\" + img, T_images_path + img)
        shutil.copyfile(label_path + "\\" + img[:-len(img_tail[1])] + label_tail,
                        T_labrls_path + img[:-len(img_tail[1])] + label_tail)
    else:
        shutil.copyfile(image_path + "\\" + img, F_images_path + img)
