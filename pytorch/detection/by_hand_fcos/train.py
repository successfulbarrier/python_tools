# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   train.py
# @Time    :   2023/12/12 17:00:27
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   网络训练脚本
import torchvision.models as models

from dataset import my_data_load
from model import FCOS

def train1():
    dataloder = my_data_load("H:\\code\\datasets\\TS_dataset4", mode="train")

    fcos = models.detection.fcos.fcos_resnet50_fpn(pretrained=True)
    fcos.eval()
    for i, (img, label) in enumerate(dataloder):   # (img, anns)
        img = img/255
        pre = fcos(img)
        print(" ")

def train2():
    dataloder = my_data_load("H:\\code\\datasets\\TS_dataset4", mode="train")

    fcos = FCOS()
    fcos.train()
    for i, (img, label) in enumerate(dataloder):   # (img, anns)
        img = img/255
        pre = fcos(img)
        print(" ")


if __name__ == '__main__':
    train2()


