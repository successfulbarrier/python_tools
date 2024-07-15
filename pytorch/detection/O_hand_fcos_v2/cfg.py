# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   cfg.py
# @Time    :   2023/12/17 17:33:06
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   配置文件


"""
    模型配置类
"""
class DefaultConfig():
    #backbone
    backbone_name="resnet18"
    pretrained=True
    freeze_stage_1=False
    freeze_bn=True

    #fpn
    fpn_out_channels=256
    use_p5=True
    
    #head
    # class_num是更换数据集一定要改的
    class_num=20
    
    # 下面配置一般不改它
    use_GN_head=True
    prior=0.01
    add_centerness=True
    cnt_on_reg=True

    #training
    strides=[8,16,32,64,128]
    limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]

    #inference
    score_threshold=0.05
    nms_iou_threshold=0.6
    max_detection_boxes_num=1000