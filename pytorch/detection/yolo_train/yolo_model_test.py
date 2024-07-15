# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   yolo_model_load.py
# @Time    :   2023/11/14 21:39:29
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   模型测试文件

from yolo_model_load.yolo import Model
import yaml

'''	
    此文件功能:测试通过配置文件建立模型
'''	
    
 
# main函数
if __name__ == '__main__':
    cfg = "H:\\code\\python_tools\\pytorch\\detection\\yolo_train\\yolo_model_load\\yolov5l.yaml"
    hyp = "H:\\code\\python_tools\\pytorch\\detection\\yolo_train\\yolo_dataset_load\\hyp.scratch.yaml"
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    model = Model(cfg , ch=3, nc=80, anchors=hyp.get('anchors'))

