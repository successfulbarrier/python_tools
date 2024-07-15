#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:python_tools
# author:机灵巢穴_WitNest
# datetime:2023/10/14 12:37
# software: PyCharm

"""
    分析csv文件的实验结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def mean():
    # 求取 准确率，召回率，mAP50和mAP50-95的后 n 个实验结果的均值
    n = 20
    data = pd.read_csv('D:/我的文件/科研文件/实验数据/TS实验结果_v1/yolov8x_1/results.csv')
    np_data = np.array(data)
    a = np_data[-n:, 4:8]
    a_mean = np.mean(a, axis=0)
    print("p: "+str(a_mean[0])+"  r: "+str(a_mean[1])+"  mAP50: "+str(a_mean[2])+"  map50-95: "+str(a_mean[3]))


if __name__ == '__main__':
    mean()