# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   continuous_processing.py
# @Time    :   2024/06/18 18:54:56
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   连续处理数据

import subprocess

# 第一步，筛选标注的数据
dataset_root = "H:\\LHT\\数据集原始数据\\light_dataset\\new_add_data"
subprocess.run(["python", "./data_set/data_set_extract.py", 
                "--dataset_root", dataset_root])

# 修改汉语名称
image_path = dataset_root+"\\new_dataset\\T_images\\"
label_path = dataset_root+"\\new_dataset\\T_labels\\"
subprocess.run(["python", "./data_set/dataset_name_change.py", 
                "--image_path", image_path, 
                "--label_path", label_path])

# 转化为txt
json_path = dataset_root+"\\new_dataset\\T_labels\\"
txt_path = dataset_root+"\\new_dataset\\txt_labels\\"
names_path = "H:\\LHT\\数据集原始数据\\light_dataset\\classes.names"
subprocess.run(["python", "./data_set/json_txt.py", 
                "--json_path", json_path, 
                "--txt_path", txt_path, 
                "--names_path", names_path])

# 划分数据集，有些参数脚本中设置
image_path = dataset_root+"\\new_dataset\\T_images\\"
label_path = dataset_root+"\\new_dataset\\txt_labels\\"
subprocess.run(["python", "./data_set/dataset_balance_split.py", 
                "--dataset_root", dataset_root,
                "--image_path", image_path, 
                "--label_path", label_path, 
                "--names_path", names_path])
