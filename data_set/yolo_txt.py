# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   yolo_txt.py
# @Time    :   2023/12/27 15:41:13
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   将yolo格式的数据集使用txt文件进行划分

import os

def write_relative_paths_to_txt(images_dir, output_file, root_path):
    with open(output_file, 'w') as file:
        for root, dirs, files in os.walk(images_dir):
            for filename in files:
                file.write("./" + os.path.relpath(os.path.join(root, filename), root_path).replace("\\","/") + '\n')


if __name__ == '__main__':
    write_relative_paths_to_txt('H:\\code\\datasets\\VisDrone\\val\\images', 
                                'H:\\code\\datasets\\VisDrone\\val\\val.txt',
                                "H:\\code\\datasets\\VisDrone\\val")