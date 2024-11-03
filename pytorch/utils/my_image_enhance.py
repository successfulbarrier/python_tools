#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:Dive_into_deep_learning_2.0
# author:机灵巢穴_WitNest
# datetime:2023/9/7 8:54
# software: PyCharm

import torchvision
from PIL import Image

img = Image.open("D:\my_file\Python_project\my_learn_project\demo_opencv\images\mao.png")
img.show()

# 调整图像大小
image_transform1 = torchvision.transforms.Resize([100, 100])

# 灰度变换
image_transform2 = torchvision.transforms.Grayscale(3)

# 规范化
image_transform3 = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5)),
    torchvision.transforms.ToPILImage()
    ])

# 随机旋转
image_transform4 = torchvision.transforms.RandomRotation([0, 90])   # 0-90度之间随机旋转角度

# 中心裁剪
image_transform5 = torchvision.transforms.CenterCrop([100, 100])

# 随机裁剪
image_transform6 = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))  # 输出图片的尺寸(200, 200), 裁剪原始图片的比例：scale=(0.1, 1), 裁剪区域高宽比：ratio=(0.5, 2)

# 高斯模糊
image_transform7 = torchvision.transforms.GaussianBlur(7)   # 核大小 3*3

# 随机改变图像颜色
image_transform8 = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5,
    hue=0.5)  # 亮度：brightness=0.5（上下随机调整50%）, 对比度：contrast=0(不变), 饱和度：saturation=0, 色调：hue=0

# 翻转图像
image_transform9 = torchvision.transforms.RandomHorizontalFlip()  # 左右翻转
image_transform10 = torchvision.transforms.RandomVerticalFlip()  # 上下翻转


img2 = image_transform10(img)
img2.show()

image_transform = torchvision.transforms.Compose([image_transform9, image_transform8,
                                                       image_transform6, torchvision.transforms.ToTensor()])


