# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   test03.py
# @Time    :   2024/04/21 19:23:12
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   新创建的python文件

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn as nn


# 读取图片
img = Image.open('test/1.jpg')

# 转换为tensor类型
transform = transforms.ToTensor()
img_tensor = transform(img)

# 将图片转换为numpy数组
img_np = np.array(img)

# 使用PyTorch进行傅里叶变换
img_fft = torch.fft.fftn(torch.tensor(img_np)).permute(2, 0, 1).unsqueeze(0)

# 进行实部和虚部的卷积
conv_layer1 = nn.Conv2d(3, 12, 3, stride=1, padding=0)
conv_layer2 = nn.Conv2d(3, 12, 3, stride=1, padding=0)

conv_real = conv_layer1(img_fft.real)
conv_imag = conv_layer2(img_fft.imag)

# 合并实部和虚部
combined = torch.complex(conv_real, conv_imag)

# 进行傅里叶逆变换
img_ifft = torch.fft.ifftn(combined)

print(img_fft.shape)
print(img_ifft.shape)
