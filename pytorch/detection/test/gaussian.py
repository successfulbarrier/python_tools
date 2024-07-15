# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   gaussian.py
# @Time    :   2023/12/24 19:52:46
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   新创建的python文件


import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def gaussian_2d(shape, sigma_x=1, sigma_y=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_truncate_gaussian(heatmap, center, h_radius, w_radius, k=1):
    h, w = 2 * h_radius + 1, 2 * w_radius + 1
    sigma_x = w / 6
    sigma_y = h / 6
    gaussian = gaussian_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
    gaussian = heatmap.new_tensor(gaussian)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, w_radius), min(width - x, w_radius + 1)
    top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[h_radius - top:h_radius + bottom,
                        w_radius - left:w_radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def test1():
    # 生成目标框中点的高斯模糊数据
    heatmap = torch.zeros(80, 80)
    cent = torch.tensor([[10, 10],[20, 20]])
    h = torch.tensor([5, 5])    # 纵向
    w = torch.tensor([7, 7])    # 横向
    draw_truncate_gaussian(heatmap, cent[0], h[0].item(), w[0].item())
    ax = sns.heatmap(heatmap)
    plt.show()  

if __name__ == '__main__':
    # 生成目标框中点的高斯模糊数据
    heatmap = torch.zeros(80, 80)
    cent = torch.tensor([[10, 10],[20, 20]])
    h = torch.tensor([5, 5])    # 纵向
    w = torch.tensor([7, 7])    # 横向
    draw_truncate_gaussian(heatmap, cent[0], h[0].item(), w[0].item())
    # ax = sns.heatmap(heatmap)
    
    # 生成表标签框的坐标信息映射到对应正样本的位置
    box_target_inds = heatmap > 0
    box_target = torch.ones((4, 80, 80)) * -1
    box = torch.tensor([10, 16, 12, 17]).float()
    box_target[:, box_target_inds] = box[:,None]  # 第一个通道需要对齐,矩阵数据类型也要相同
    # ac = sns.heatmap(box_target[2])
    
    # 生成和面积相关的权重
    area = 1000   # 假定一个标签框的面积为10
    local_heatmap = heatmap[box_target_inds]    # 取出对应位置的数据
    ct_div = local_heatmap.sum()    # 用于平衡乘以面积之后过大
    local_heatmap *= area
    reg_weight = torch.zeros(80, 80)
    reg_weight[box_target_inds] = local_heatmap / ct_div
    bx = sns.heatmap(reg_weight)
    plt.show()  

