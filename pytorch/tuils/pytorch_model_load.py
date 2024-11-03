# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   pytorch_model_load.py
# @Time    :   2023/12/19 17:02:41
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   使用pytorch官方提供的分类网络作为检测网络的主干网络

import torch
import torchvision.models as models
import torch.nn as nn


def test01():
    # 加载预训练的 resnet50 模型
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # 移除最后一层全连接层
    backbone = nn.Sequential(*list(backbone.children())[:-2])

    print(backbone)

    # 随机一个输入
    input_data = torch.randn(1, 3, 640, 640, dtype=torch.float32)
    print(input_data.shape)

    # 获取网络的多层输出
    out1 = backbone[:-2](input_data)
    out2 = backbone[-2](out1)
    out3 = backbone[-1](out2)
    print(out1.shape)
    print(out2.shape)
    print(out3.shape)


class ResNet(nn.Module):
    def __init__(self, size="resnet50") -> None:
        super(ResNet, self).__init__()
        # 加载预训练的 resnet50 模型
        if size == "resnet18":
            # 网上下载权重 weights=models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=None)
            self.backbone.load_state_dict(torch.load('./checkpoint/resnet34.pth'),strict=True)
        elif size == "resnet34":
            self.backbone = models.resnet34(weights=None)
            self.backbone.load_state_dict(torch.load('./checkpoint/resnet34.pth'),strict=True)
        elif size == "resnet50":
            self.backbone = models.resnet50(weights=None)
            self.backbone.load_state_dict(torch.load('./checkpoint/resnet50.pth'),strict=True)
        elif size == "resnet101":
            self.backbone = models.resnet101(weights=None)
            self.backbone.load_state_dict(torch.load('./checkpoint/resnet101.pth'),strict=True)
        elif size == "resnet152":
            self.backbone = models.resnet152(weights=None)
            self.backbone.load_state_dict(torch.load('./checkpoint/resnet152.pth'),strict=True)
        else:
            raise Exception("resnet网络size设置错误！！！！")   # 抛出异常
            
        # 移除最后一层全连接层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])   # 具体如何裁剪要看具体模型和自己需求

        # 计算输出层的通道数
        self.out_channel = [0, 0, 0]
        self.out_channel[0], self.out_channel[1], self.out_channel[2] = self.get_out_channel()

    def forward(self, x):
        # 获取网络的多层输出
        out1 = self.backbone[:-2](x)
        out2 = self.backbone[-2](out1)
        out3 = self.backbone[-1](out2)
        return (out1, out2, out3)
    
    def freeze_bn(self):    # 冻结BN层
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                # print(layer)
                layer.eval()

    def get_out_channel(self):
        input_data = torch.randn(1, 3, 640, 640, dtype=torch.float32)
        (out1, out2, out3) = self.forward(input_data)
        return out1.shape[1], out2.shape[1], out3.shape[1]


if __name__ == '__main__':
    # test01()
    resnet50 = ResNet(size="resnet152")
    print(resnet50.out_channel[0])
    print(resnet50.out_channel[1])
    print(resnet50.out_channel[2])