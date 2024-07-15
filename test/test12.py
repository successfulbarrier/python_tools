# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   test12.py
# @Time    :   2024/05/27 17:32:49
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   新创建的python文件


# 绘制一个有15个数据的折线图并设定x轴和y轴的数据
import numpy as np
import matplotlib.pyplot as plt

# 生成15个随机数据作为示例
Y = [0.6601, 0.6613, 0.6632, 0.6655, 0.6675, 0.6704, 0.6732, 0.6757, 0.6769, 0.6777, 0.6783, 0.6779, 0.6771]

# 设置x轴数据
X = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]

# 绘制折线图
plt.plot(X, Y, color='blue', marker='^')
plt.xlabel('ratio')
plt.ylabel('map50')
plt.title('lnner_iou')
plt.legend()
plt.show()
