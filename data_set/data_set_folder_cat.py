# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   data_set_folder_cat.py
# @Time    :   2023/11/25 19:58:49
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   将多个文件夹的文件合并到一个文件夹

import os
import shutil
from tqdm import tqdm


def merge_folders(folder_list, destination_folder):
    """
    将一个列表文件夹中的文件合并到一个文件夹中
    """
    for folder in tqdm(folder_list):
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            if os.path.isfile(item_path):
                shutil.move(item_path, destination_folder)
    
    
# main函数
if __name__ == '__main__':
    folder_list1 = ["H:\\实验数据\\TS_dataset2\\硅胶变色_label",
                    "H:\\实验数据\\TS_dataset2\\违章作业_label",
                    "H:\\实验数据\\TS_dataset2\\箱门闭合状态_label",
                    "H:\\实验数据\\TS_dataset2\\压板状态_label"]
    destination_folder1 = "H:\\实验数据\\TS_dataset2\\jsons2"
    merge_folders(folder_list1, destination_folder1)
