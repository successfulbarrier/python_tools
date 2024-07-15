#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:Dive_into_deep_learning_2.0
# author:机灵巢穴_WitNest
# datetime:2023/8/31 9:12
# software: PyCharm

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2 as cv
import torch
import matplotlib.pyplot as plt
import torch.nn as nn


"""
    绘制单条曲线
"""


def test_01():
    writer = SummaryWriter()
    for x in range(1, 101) :
        y = 2 * x
        writer.add_scalar('y = 2x', y, x)   # "y = 2x" 图名称，
    writer.close()


"""
    绘制多条曲线
"""


def test_02():
    writer = SummaryWriter()
    r = 5
    for x in range(1, 101):
        writer.add_scalars('run_14h', {'xsinx': x * np.sin(x / r),
                                       'xcosx': x * np.cos(x / r),
                                       'xtanx': x * np.tan(x / r)}, x)
    writer.close()


"""
    绘制直方图
"""


def test_03():
    writer = SummaryWriter()
    for step in range(10):
        x = np.random.randn(1000)
        writer.add_histogram('直方图', x, step)
    writer.close()


"""
    显示图片
"""


def test_04():
    writer = SummaryWriter()
    img = cv.imread('D:/my_file/Python_project/my_learn_project/demo_opencv/images/2.jpg', cv.IMREAD_COLOR)  # 输入图像要是3通道的，所以读取彩色图像
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = torch.tensor(img.transpose(2, 0, 1))  # cv读取为numpy图像为(H * W * C)，所以要进行轴转换
    writer.add_image('img', img, 0)
    writer.close()


"""
    渲染,可以直接显示matplotlib绘制的图像
"""


def test_05():
    writer = SummaryWriter()

    x = np.linspace(0, 10, 1000)
    y = np.sin(x)

    figure1 = plt.figure()
    plt.plot(x, y, 'r-')
    writer.add_figure('my_figure', figure1, 0)
    writer.close()


"""
    显示网络结构
"""


def test_06():
    writer = SummaryWriter()

    # 定义网络
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        nn.Sigmoid(),
        nn.Linear(84, 10))

    x = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    writer.add_graph(net, x)
    writer.close()


"""
    嵌入,存在问题
"""


def test_07():
    writer = SummaryWriter()
    a = list(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"])
    embedded = torch.randn(16, 3)
    writer.add_embedding(embedded, metadata=a)
    writer.close()


"""
    tensorboard运行方式
    cd到你的py文件目录下，输入 tensorboard --logdir runs --port=6007
    打开对应网址
"""
if __name__ == '__main__':
    test_01()