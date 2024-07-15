# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   dataset_name_change.py
# @Time    :   2023/11/19 21:49:18
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   修改文件名


import os
import re
from tqdm import tqdm

import random
import string
import argparse

# 产生随机字符串
def generate_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

# 检查字符串中是否存在中文
def contains_chinese(text):
    return bool(re.search('[\u4e00-\u9fff]', text))


if __name__ == '__main__':
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description='Process some integers.')

    # 传递带值的参数
    parser.add_argument('--image_path', type=str, 
                        default="H:\\LHT\\数据集原始数据\\light_dataset\\new_add_data\\new_dataset\\T_images\\", help='name of the user')
    parser.add_argument('--label_path', type=str, 
                        default="H:\\LHT\\数据集原始数据\\light_dataset\\new_add_data\\new_dataset\\T_labels\\", help='name of the user')
    
    # 解析参数
    args = parser.parse_args()
    
    # image_path = "H:\\LHT\\数据集原始数据\\light_dataset\\new_add_data\\new_dataset\\T_images\\"
    # label_path = "H:\\LHT\\数据集原始数据\\light_dataset\\new_add_data\\new_dataset\\T_labels\\"

    images = os.listdir(args.image_path)
    labels = os.listdir(args.label_path)
    a = 0
    for img in tqdm(images):
        if contains_chinese(img):
            new_name = generate_random_string(8)+str(a)
            a +=1
            os.rename(args.image_path+img, args.image_path+new_name+".jpg")
            os.rename(args.label_path+img[:-4]+".json", args.label_path+new_name+".json")

