#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:python_tools
# author:机灵巢穴_WitNest
# datetime:2023/10/28 12:15
# software: PyCharm

"""
    删除数据集里面的一个图片和对应的标签文件
"""

import os


image_path = "D:\\我的文件\\科研文件\\实验数据\\light_dataset\\images\\"
label_path = "D:\\我的文件\\科研文件\\实验数据\\light_dataset\\json\\"

file_name = "f2875ab2f0519291ee7efd5a3149364"          # 文件名不要后缀

image_path = image_path + file_name + ".jpg"
label_path = label_path + file_name + ".json"

if os.path.exists(image_path):
    os.remove(image_path)

if os.path.exists(label_path):
    os.remove(label_path)

print("以删除文件！！！！")