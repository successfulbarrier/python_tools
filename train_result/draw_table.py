# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   draw_table.py
# @Time    :   2024/07/03 15:19:31
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   绘制表格


from tabulate import tabulate

# 数据
data = [
    ["Name", "Age", "City"],
    ["Alice", 24, "New York"],
    ["Bob", 27, "Los Angeles"],
    ["Charlie", 22, "Chicago"]
]

# 使用tabulate绘制表格
table = tabulate(data, headers="firstrow", tablefmt="grid")

print(table)

