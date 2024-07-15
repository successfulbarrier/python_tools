# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   cross_entropy.py
# @Time    :   2024/01/01 21:46:47
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   新创建的python文件
import torch
import torch.nn.functional as F

def test():
    a = torch.tensor([[0.8,1.2,14],
                     [1,1,0.5]],dtype=torch.float32)
    b = torch.tensor([0,1])
    c = F.cross_entropy(a, b, reduction='none')
    print(c)


if __name__ == '__main__':
    test()