# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   png_jpg.py
# @Time    :   2024/07/21 17:02:50
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   将png图片转化为jpg图片
import os
from PIL import Image

path = "/media/lht/D_Project/datasets/light_dataset2/train/images"

# 获取目录下所有文件
files = os.listdir(path)

# 将png图片转化为jpg图片
for file in files:
    if file.endswith(".png"):
        img = Image.open(os.path.join(path, file))
        img = img.convert("RGB")
        img.save(os.path.join(path, file.replace(".png", ".jpg")))
        os.remove(os.path.join(path, file))

print("图片格式转换完成！")


