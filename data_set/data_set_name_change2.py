# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   data_set_name_change2.py
# @Time    :   2024/07/23 21:45:23
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   重新修改图片和标签的名字，并添加版本前缀
import os
import argparse
import random
import string
from tqdm import tqdm

#-------------------------------------------------#
#   获取参数
#-------------------------------------------------#
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--images_path', type=str, 
                    default="/media/lht/LHT/light/现场数据/图片", 
                    help='name of the user')
parser.add_argument('--labels_path', type=str, 
                    default="/media/lht/LHT/light/现场数据/labels", 
                    help='name of the user')
parser.add_argument('--version', type=str, 
                    default="v2", 
                    help='name of the user')
args = parser.parse_args()



#-------------------------------------------------#
#   重命名文件
#-------------------------------------------------#
def rename_files(images_path, labels_path, version):
    image_files = os.listdir(images_path)
    label_files = os.listdir(labels_path)

    for img_file in tqdm(image_files):
        for label_file in label_files:
            if img_file.split('.')[0] == label_file.split('.')[0]:
                new_name = version + '_' + ''.join(random.choices(string.ascii_letters + string.digits, k=15))
                os.rename(os.path.join(images_path, img_file), os.path.join(images_path, new_name + '.' + img_file.split('.')[-1]))
                os.rename(os.path.join(labels_path, label_file), os.path.join(labels_path, new_name + '.' + label_file.split('.')[-1]))
            else:
                assert Exception("未找到对应的标签文件！！！")
            
rename_files(args.images_path, args.labels_path, args.version)

