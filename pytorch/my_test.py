#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:Dive_into_deep_learning_2.0
# author:机灵巢穴_WitNest
# datetime:2023/8/30 17:31
# software: PyCharm
from collections import OrderedDict
import torch
from torch import nn
import torchvision
from d2l import torch as d2l


"""
    定义模型
"""
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))


"""
    验证模型，并查看每层的输出矩阵形状
"""
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)


"""
    生成随机矩阵（一下生成的矩阵元素顺序均是随机，但是分布不同）
"""
# torch.Tensor(4, 4)  # 随机生成浮点数
# torch.rand(4, 4)    # 0-1 之间均匀分布
# torch.randn(4, 4)   # 0-1 之间正态分布
# torch.randint(1, 10, (4, 4))    # 生成指定范围内的整数


# model = torchvision.models.resnet18()
"""
    获取网络各层的参数
"""
# 法一
# i = 1
# for name, params in net.named_parameters():
#     print("----------------第{}层---------------".format(i))
#     print(name)
#     print(params.data) # 获取纯数据
#     print(params.shape)
#     i += i

# 法二，将参数存储在列表之中
# params = list(net.named_parameters())
# print(len(params))
# print(params[0])

# 法三
# i = 1
# for k, v in net.state_dict().items(): # items()返回可遍历的(键, 值) 元组数组
#     print("----------------第{}层---------------".format(i))
#     print(k)
#     # print(v) # 纯数值
#     print(v.shape)
#     i += i

# 法四，返回的是网络定义信息
# for x in net.children():
#     print(x)


"""
    获取网络的每层输出
"""


class IntermediateLayerGetter(nn.ModuleDict):
    """ get the output of certain layers """

    def __init__(self, model, return_layers):
        # 判断传入的return_layers是否存在于model中
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}  # 构造dict
        layers = OrderedDict()
        # 将要从model中获取信息的最后一层之前的模块全部复制下来
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)  # 将所需的网络层通过继承的方式保存下来
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        # 将所需的值以k,v的形式保存到out中
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


# 如果没有层名称输入层号即可 ”需要输出的层号“：“在读取出来的字典中对应的名称”
return_layers = {'0': '0', '1': '1', '2': '2', '3': '3'}
backbone = IntermediateLayerGetter(net, return_layers)

backbone.eval()
x = torch.randn(1, 1, 28, 28)
out = backbone(x)
print(out['0'].shape, out['1'].shape, out['2'].shape, out['3'].shape)
print(out['0'])


"""
    图像增广，使用torchvision提供的方法，可以减少过拟合。
"""


def test_transform():
    # 翻转图像
    image_transform1 = torchvision.transforms.RandomVerticalFlip()  # 上下翻转
    image_transform12 = torchvision.transforms.RandomHorizontalFlip()   # 左右翻转
    # 随机裁剪
    image_transform2 = torchvision.transforms.RandomResizedCrop(
        (200, 200), scale=(0.1, 1), ratio=(0.5, 2))     # 输出图片的尺寸(200, 200), 裁剪原始图片的比例：scale=(0.5, 1), 裁剪区域高宽比：ratio=(0.5, 2)
    # 随机改变图像亮度
    image_transform3 = torchvision.transforms.ColorJitter(
        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)    # 亮度：brightness=0.5（上下随机调整50%）, 对比度：contrast=0(不变), 饱和度：saturation=0, 色调：hue=0
    # 一般不单独使用一种图像增广方式，而是联合起来使用,翻转-变色-裁剪。
    image_transform4 = torchvision.transforms.Compose([image_transform1, image_transform3,
                                                       image_transform2, torchvision.transforms.ToTensor()])


if __name__ == '__main__':
    a = torch.randn(4, 4)
    a.unflatten(0, 1)