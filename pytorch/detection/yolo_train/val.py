# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   val.py
# @Time    :   2023/11/26 18:47:40
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   自己写的yolo验证方法

# 官方库
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import optim
import math

# 自己实现的库
from cfg import Cfg
from utils import load_weight


# 三方库
import yolo_dataset_load.datasets as ds
from torch.optim import SGD, Adam, AdamW
from yolo_dataset_load.general import labels_to_class_weights
from yolo_dataset_load.torch_utils import select_device
from yolov8_model_load.yolov8 import Model_v8
from yolov8_loss_load.loss import v8DetectionLoss    
from yolo_val.val import val_my   

 
def v0():
    """
        最简单的训练流程只计算了损失，使用GPU, yolov8模型,验证
    """
    cfg = Cfg("test.yaml", "H:/code/python_tools/pytorch/detection/yolo_train/")
    cfg.device = select_device(cfg.device, cfg.batch)
    val_loader, _ = ds.create_dataloader_train(cfg.dataset_val, cfg.cfg_hyp["imgsz"], 2,
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht_val:")
    model = Model_v8(cfg.cfg_model , ch=cfg.ch, nc=cfg.nc).to(cfg.device)
    # 加载模型
    load_weight(cfg.cfg_weight, model, train=False)

    # 传递模型参数
    model.nc = cfg.nc  # attach number of classes to model
    model.hyp = cfg.cfg_hyp  # attach hyperparameters to model
    model.names = cfg.names
    compute_loss = v8DetectionLoss(model)
    compute_loss.device = cfg.device
    result, _ = val_my(val_loader, model, compute_loss, nc=cfg.nc)
    # print('mp', result[0]) 
    # print('mr', result[1])
    # print('map50', result[2])
    # print('map50-95', result[3])
    # print('val_lbox', result[4])
    # print('val_lcls', result[5])
    # print('val_ldfl', result[6]) 

    
# main函数
if __name__ == '__main__':
    v0()