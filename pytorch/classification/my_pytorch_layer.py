# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   my_pytorch_layer.py
# @Time    :   2023/11/25 22:34:30
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   新创建的python文件

import torch
import torch.nn as nn


class CustomConv2d(nn.Module):
    """
        手动实现2d卷积层
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CustomConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        output = torch.zeros(input.size(0), self.out_channels, (input.size(2) + 2 * self.padding - self.kernel_size) // self.stride + 1, (input.size(3) + 2 * self.padding - self.kernel_size) // self.stride + 1)
        for i in range(output.size(2)):
            for j in range(output.size(3)):
                output[:, :, i, j] = torch.sum(input[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size].unsqueeze(1) * self.weight.unsqueeze(2).unsqueeze(3), dim=(2, 3)) + self.bias
        return output