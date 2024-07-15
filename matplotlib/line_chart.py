#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:demo_Matplotlib
# author:全栈学习者
# datetime:2023/5/27 12:27
# software: PyCharm
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-whitegrid')


# 绘制最简单的折线图
def test01():
    fig = plt.figure()  # 创建图像对象
    ax = plt.axes()  # 创建维度对象

    x = np.linspace(0, 10, 1000)  # 生成一组数据
    ax.plot(x, np.sin(x), "--c", label='sin(x)')  # 绘制图像
    ax.plot(x, np.cos(x), label='cos(x)')  # 绘制图像
    plt.axis('tight')
    plt.legend()
    fig.show()


# 绘制最简单的散点图
def test02():
    fig = plt.figure()  # 创建图像对象
    ax = plt.axes()  # 创建维度对象
    x = np.linspace(0, 10, 30)
    y = np.sin(x)
    ax.plot(x, y, 'o', color='black', label='sin(x)')
    ax.axis('tight')
    ax.legend()
    fig.show()


# 使用plt.scatter()函数绘制散点图
def test03():
    rng = np.random.RandomState(0)
    x = rng.randn(100)
    y = rng.randn(100)
    colors = rng.rand(100)
    sizes = 1000 * rng.rand(100)

    plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,
                cmap='viridis')  # alpha关键字参数对点的透明度进行了调整
    plt.colorbar()  # 显示颜色对比条
    plt.show()


# 绘制误差条
def test04():
    x = np.linspace(0, 10, 50)
    dy = 0.8
    y = np.sin(x) + dy * np.random.randn(50)

    plt.errorbar(x, y, yerr=dy, fmt='.k');
    plt.show()


if __name__ == '__main__':
    test04()
