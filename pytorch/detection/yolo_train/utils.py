# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   utils.py
# @Time    :   2023/11/15 19:18:48
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   在编程过程中用到的一些小函数
import torch
import math
import numpy as np  
from tqdm import tqdm

###############################################################
def load_weight(weight_path=None, model=None, train=True):
    """
        加载权重文件,加载pth文件
    """
    if weight_path:
        ckpt = torch.load(weight_path)
        if False:   # 模型配置字典
            new_ckpt_k = list(ckpt)
            new_ckpt_v = list(ckpt.values())
            for i, k in enumerate(new_ckpt_k):
                new_ckpt_k[i] = k[6:]
            ckpt = dict(zip(new_ckpt_k, new_ckpt_v))
            ckpt = torch.load(ckpt).float().state_dict() # 存在问题
        if train:
            del_key = []
            for key, _ in ckpt.items():
                if ("22" in key):
                    del_key.append(key)
            for key in del_key:
                del ckpt[key]                    
        mk, _ = model.model.load_state_dict(ckpt, strict=False) 
        print("\033[32m" + "----> 未加载的参数：" + str(mk) + "\033[0m")
    else:
        print("\033[32m" + "----> 未加载权重！！！" + "\033[0m")
    

#############################################################
def non_max_suppression(boxes, scores, threshold):
    """
    非极大值抑制算法
    Args:
    - boxes: 边界框坐标，形状为(N, 4)，N为边界框数量，每个边界框由左上角和右下角坐标表示
    - scores: 每个边界框的置信度得分，形状为(N,)
    - threshold: 重叠阈值，大于该阈值的边界框会被抑制

    Returns:
    - keep: 保留的边界框索引
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        overlap = (w * h) / areas[order[1:]]

        inds = np.where(overlap <= threshold)[0]
        order = order[inds + 1]

    return keep


def my_val(dataloader, model, compute_loss,nc=80):
    device = next(model.parameters()).device
    names = model.names
    iouv = torch.linspace(0.5, 0.95, 10, device=device) # 产生map50-95的10个值最后取均值
    niou = iouv.numel() # 计算iouv的长度
    
    # 定义进度条样式
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}')
    model.eval()    # 退出训练模式，不更新参数
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        # 数据预处理
        im = im.to(device).float()
        im /= 255
        nb, _, height, width = im.shape
        targets = targets.to(device)

        # 前向传播
        preds, train_out = model(im)
        im = im.cpu()
        
        # 损失计算
        loss = torch.zeros(3, device=device)
        loss += compute_loss(train_out, targets)[1]     # 分别获取三个损失然后累加，最后求平均        


# main函数
if __name__ == '__main__':
    iouv = torch.linspace(0.5, 0.95, 10) # 产生map50-95的10个值最后取均值
    niou = iouv.numel()
    print(niou)
    

