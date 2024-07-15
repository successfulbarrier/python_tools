# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   export_onnx.py
# @Time    :   2023/12/18 12:59:59
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   导出onnx模型


import torch
import torchvision.models as models

# 加载预训练模型
model = models.resnet18(pretrained=True)

# 将模型转换为eval模式
model.eval()

# 创建一个示例输入
example_input = torch.rand(1, 3, 224, 224)

# 导出模型为ONNX格式
torch.onnx.export(model, example_input, "resnet18.onnx", verbose=True)