#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:python_tools
# author:机灵巢穴_WitNest
# datetime:2023/10/28 20:15
# software: PyCharm

"""
    将 classes.txt 文件中的类别信息转化为 python 字典和 yaml文件
"""

import yaml

file_path = "J:\\实验数据\\light_dataset\\classes.names"
class_label1 = {'light_off_red': 0, 'light_on_red': 1, 'light_off_green': 2, 'light_on_green': 3,
                'light_off_yellow': 4, 'light_on_yellow': 5, 'light_off_white': 6, 'light_on_white': 7,
                'switch_one_0': 8, 'switch_one_270': 9, 'switch_two_0': 10, 'switch_two_270': 11, 'switch_three_0': 12,
                'switch_three_270': 13, 'switch_four_0': 14, 'switch_four_270': 15, 'switch_five_0': 16,
                'switch_five_270': 17, 'ya_ban_off': 18, 'ya_ban_on': 19, 'group_red': 20, 'group_green': 21}
names = {}
num = 0

with open(file_path, 'r') as f:
    while True:
        line = f.readline()
        if line == '':  # 如果文件结束，结束循环
            break
        class_label1[line[:-len("\n")]] = num
        names[num] = line[:-len("\n")]
        num += 1

print(class_label1)
print(names)

# 保存为yolo数据集配置文件
dataset_label = {"path": "", "train": "", "val": "", "test": "", "names": names}
with open('./文件/dataset_label.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(data=dataset_label, stream=f, allow_unicode=True)