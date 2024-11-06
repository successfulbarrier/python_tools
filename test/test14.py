# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   test14.py
# @Time    :   2024/11/06 18:43:44
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   新创建的python脚本


import torch
from torch import nn
tokens = torch.rand(4, 32, 512)
valid_lens = torch.tensor([31, 24, 16, 8])  # 输入序列的有效长度
# valid_lens转化填充位置为掩码
valid_mask = torch.zeros((valid_lens.size(0), tokens.shape[1]), dtype=torch.bool)  # 初始化全为0的二维 Tensor
for i in range(valid_lens.size(0)):
    valid_len_value = int(valid_lens[i].item())
    valid_mask[i, valid_len_value:] = 1  # 将前 valid_len 个元素设为 1
valid_mask = valid_mask.to(tokens.device)
print(valid_mask.shape)
transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(512, 8, batch_first=True), num_layers=6)
memory = transformer_encoder(tokens, src_key_padding_mask=valid_mask)
print(memory.shape)