# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   result_csv_draw.py
# @Time    :   2023/11/14 10:30:34
# @Author  :   lht 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   新创建的python文件


"""
    绘制保存的result.csv中的结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def draw_all():
    # 求取 准确率，召回率，mAP50和mAP50-95的后 n 个实验结果的均值
    data = pd.read_csv('H:\\code\\yolov8_ubuntu_G\\ultralytics\\yolo\\v8\detect\\runs\detect\\train30\\results.csv')
    # print(data.columns)
    # print(data.index)
    num_columns = len(data.columns)
    gs = gridspec.GridSpec(num_columns, 2, height_ratios=[1]*num_columns)  # 创建一个网格，每个子图的高度比例都是1

    for i, column in enumerate(data.columns):
        ax = plt.subplot(gs[i])  # 在网格上创建子图
        column_data = data[column]
        ax.plot(column_data, label=column)  # 绘制折线图
        ax.set_title(column)  # 设置子图的标题
        ax.set_xlabel('X Label')  # 设置x轴的标签
        ax.set_ylabel('Y Label')  # 设置y轴的标签

    plt.tight_layout()  # 自动调整子图的位置
    plt.show()  # 显示图像


def draw():
    # 求取 准确率，召回率，mAP50和mAP50-95的后 n 个实验结果的均值
    data = pd.read_csv('H:\\code\\yolov8_ubuntu_G\\ultralytics\\yolo\\v8\detect\\runs\detect\\train30\\results.csv')
    # print(data.columns)
    # print(data.index)
    num_columns = len(data.columns)
    
    for column in data.columns[1:11]:
        # 使用matplotlib库绘制折线图
        column_data = data[column]
        plt.plot(column_data)
        plt.title(column)  # 设置图的标题
        plt.xlabel('X Label')  # 设置x轴的标签
        plt.ylabel('Y Label')  # 设置y轴的标签
        plt.show()  # 显示图像


if __name__ == '__main__':
    draw()