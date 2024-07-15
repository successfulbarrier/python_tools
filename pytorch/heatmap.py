#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:python_tools
# author:机灵巢穴_WitNest
# datetime:2023/10/23 19:22
# software: PyCharm

"""
    绘制热力图
"""

import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np

def test1():
    a = torch.randint(0, 10, (1000, 1000))
    ax = sns.heatmap(a)
    plt.show()


def test2():
    # 生成数据
    data = np.random.random((100, 100, 10))

    # 绘制3D热力图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.meshgrid(np.arange(data.shape[0]),
                        np.arange(data.shape[1]),
                        np.arange(data.shape[2]))

    ax.scatter(x, y, z, c=data.flatten(), cmap='hot')
    plt.show()


if __name__ == '__main__':
    test1()