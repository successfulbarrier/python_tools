# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   yolo_model_load.py
# @Time    :   2023/11/14 21:39:29
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   模型测试文件

from yolov8_model_load.yolov8 import Model_v8
from yolov8_loss_load.loss import v8DetectionLoss
from cfg import Cfg
'''	
    此文件功能:测试通过配置文件建立模型
'''	
    
 
# main函数
if __name__ == '__main__':
    cfg = Cfg("test.yaml")
    model = Model_v8(cfg.cfg_model , ch=3, nc=80)
    model.hyp = cfg.cfg_hyp
    loss = v8DetectionLoss(model)
