#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:python_tools
# author:机灵巢穴_WitNest
# datetime:2023/10/28 10:50
# software: PyCharm

"""
    修改labelme标注好的某一个类别的 label
"""
# coding:utf-8

import os
import json
from tqdm import tqdm

json_dir = 'D:/我的文件/科研文件/实验数据/light_dataset/json/'  # 写入json文件的文件夹路径,最后要加 /
json_files = os.listdir(json_dir)

# 写自己的旧标签名和新标签名
old_name = "light_off-white"
new_name = "light_off_white"

for json_file in tqdm(json_files):
    json_file_ext = os.path.splitext(json_file)

    if json_file_ext[1] == '.json':
        jsonfile = json_dir + json_file

        with open(jsonfile, 'r', encoding='utf-8') as jf:
            info = json.load(jf)

            for i, label in enumerate(info['shapes']):
                if info['shapes'][i]['label'] == old_name:
                    info['shapes'][i]['label'] = new_name
                    # 找到位置进行修改
            # 使用新字典替换修改后的字典
            json_dict = info

        # 将替换后的内容写入原文件
        with open(jsonfile, 'w') as new_jf:
            json.dump(json_dict, new_jf)

