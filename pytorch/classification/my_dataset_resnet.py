#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:Dive_into_deep_learning_2.0
# author:机灵巢穴_WitNest
# datetime:2023/9/3 10:10
# software: PyCharm

# import sys
# sys.path.append("..")   # 将上一级目录添加到

from torch import nn
# from d2l import torch as d2l
from torch.nn import functional as F
# from pytorch import my_lib
import torch


"""
    resnet块
"""


class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides, bias=False)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1, bias=False)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


"""
    resnet主干网络
"""


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


def model_export_onnx():
    b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(512, 10))

    net.eval()
 
    dummy_input = torch.randn(1, 3, 64, 64)
    input_names = ['data']
    output_names = ['fc']
    torch.onnx.export(net, dummy_input, 'test.onnx', 
                      export_params=True, 
                      verbose=True, 
                      input_names=input_names, 
                      output_names=output_names)


if __name__ == '__main__':
    """
            加载数据集
    """
    img_size = [224, 224]
    train_iter, test_iter, class_list = my_lib.C_data_load("D:\\my_file\\data_set\\classification\\flowers",
                                                           img_size, batch_size=64, data_enhance=[4, 4])

    """
        定义模型
    """
    b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(512, 10))

    """
        验证模型，并查看每层的输出矩阵形状
    """
    my_lib.Net_shape_print(net, (1, 3, 224, 224))

    """
        训练 
    """
    lr, num_epochs = 0.01, 20
    my_lib.C_train_my(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu(), "Momentum")