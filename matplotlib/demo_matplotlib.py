"""
    matplotlib-python编程
"""

import matplotlib.pyplot as plt
import numpy as np


# 简单创建一个图表
def test_np01():
    x = np.linspace(-3, 3, 50)
    y1 = 2*x+1
    y2 = x**2
    # 每一个 figure 就是一个窗口
    plt.figure()
    plt.plot(x, y1)
    plt.show()

    plt.figure(num=3, figsize=[8, 5])
    plt.plot(x, y2, color='blue', linewidth=1.0, linestyle='--')  # 可以设置线条的一些参数
    plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
    plt.show()


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    test_np01()