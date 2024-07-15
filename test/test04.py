# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   test03.py
# @Time    :   2024/04/21 19:23:12
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   新创建的python文件

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('test/1.jpg', 0)

# 进行傅里叶变换
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# 获取实部和虚部
magnitude_spectrum = 20 * np.log(np.abs(fshift))
phase_spectrum = np.angle(fshift)

# 显示实部和虚部
plt.subplot(121), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(phase_spectrum, cmap='gray')
plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()