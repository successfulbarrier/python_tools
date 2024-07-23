# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   jpeg_jpg.py
# @Time    :   2024/07/21 16:56:35
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   将jpeg后缀的文件修改成jpg文件
import os

path = "/media/lht/D_Project/datasets/light_dataset2/val/images"

# 获取目录下所有文件
files = os.listdir(path)

# 将jpeg后缀的文件修改成jpg后缀的文件
for file in files:
    if file.endswith(".jpeg"):
        os.rename(os.path.join(path, file), os.path.join(path, file.replace(".jpeg", ".jpg")))

print("文件后缀修改完成！")