#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:python_tools
# author:机灵巢穴_WitNest
# datetime:2023/10/28 11:05
# software: PyCharm

"""
    删除labelme标注好的某一个类别 (有BUG)
"""

import os
import json
from tqdm import tqdm

json_dir = 'D:/我的文件/科研文件/实验数据/light_dataset/json/'  # 写入json文件的文件夹路径
json_files = os.listdir(json_dir)

# 这里写你要删除的标签名
delete_name = "trip"

for json_file in tqdm(json_files):
    json_file_ext = os.path.splitext(json_file)

    if json_file_ext[1] == '.json':
        # 判断是否为json文件
        jsonfile = json_dir + json_file

        with open(jsonfile, 'r', encoding='utf-8') as jf:
            info = json.load(jf)

            for i, label in enumerate(info['shapes']):
                if info['shapes'][i]['label'] == delete_name:
                    del info['shapes'][i]
                    # 找到位置进行删除
            # 使用新字典替换修改后的字典
            json_dict = info

        # 将替换后的内容写入原文件
        with open(jsonfile, 'w') as new_jf:
            json.dump(json_dict, new_jf)

