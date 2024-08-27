# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   gray_hatmap.py
# @Time    :   2024/08/26 17:26:06
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   将灰度图像使用热力图显示

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import cv2  # 添加此行


def display_heatmap(image_path):
    # 读取RGB图像
    # 读取图像并确保其为RGB图像
    image = plt.imread(image_path)
    if len(image.shape) == 2:  # 灰度图像
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA图像
        rgb_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        rgb_image = image

    rgb_image = cv2.resize(rgb_image, (640, 640))
    
    # 检查图像是否为RGB图像
    if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
        raise ValueError("输入的图像不是RGB图像")

    # 将RGB图像转换为YCBCR图像
    ycbcr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)
    
    # 对YCBCR图像进行DCT变换,先分块再对每个块进行单独的DCT变换
    height, width = ycbcr_image.shape[0], ycbcr_image.shape[1]
    dct_image = np.zeros((height // 8, width // 8, 64 * 3), dtype=np.float32)
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            for k in range(3):
                block = ycbcr_image[i:i+8, j:j+8, k]
                dct_block = cv2.dct(np.float32(block))
                dct_image[i//8, j//8, k*64:(k+1)*64] = dct_block.flatten()
    
    # 提取B通道作为灰度图像
    # gray_image = rgb_image[:, :, 1]
    for i in range(3):
        gray_image = ycbcr_image[:, :, i]
        # 显示彩色热力图
        # 热力图颜色风格选择'viridis', 'plasma', 'inferno', 'magma'，'hot'
        plt.imshow(gray_image, cmap='viridis')
        plt.colorbar()
        plt.title('hatmap')
        plt.savefig(f'Ycbcr_{i}.png')  # 添加此行保存图片
        # plt.show()
        plt.close()
        
    for i in range(64):
        gray_image = dct_image[:, :, i]
        # 显示彩色热力图
        # 热力图颜色风格选择'viridis', 'plasma', 'inferno', 'magma'，'hot'
        plt.imshow(gray_image, cmap='viridis')
        plt.colorbar()
        plt.title('hatmap')
        plt.savefig(f'Ycbcr_dct8_{i}.png')  # 添加此行保存图片
        # plt.show()
        plt.close()
        
# 示例调用
display_heatmap('/media/lht/LHT/绘图图片/图9/原始图片.png')




