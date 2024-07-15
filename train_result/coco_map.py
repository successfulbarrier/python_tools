# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   coco_map.py
# @Time    :   2023/12/11 15:06:53
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   计算map等指标

"""
    coco评价指标的含义：具体见coco官网：https://cocodataset.org/#detection-eval
    Average Precision (AP):
        AP% AP at IoU=.50:.05:.95 (primary challenge metric) 
        APIoU=.50% AP at IoU=.50 (PASCAL VOC metric) 
        APIoU=.75% AP at IoU=.75 (strict metric)
    AP Across Scales:
        APsmall% AP for small objects: area < 322 
        APmedium% AP for medium objects: 322 < area < 962 
        APlarge% AP for large objects: area > 962
    Average Recall (AR):
        ARmax=1% AR given 1 detection per image 
        ARmax=10% AR given 10 detections per image 
        ARmax=100% AR given 100 detections per image
    AR Across Scales:
        ARsmall% AR for small objects: area < 322 
        ARmedium% AR for medium objects: 322 < area < 962 
        ARlarge% AR for large objects: area > 962
"""

# 您可以使用以下代码来使用 pycocotools 计算 COCO 数据集的 mAP：
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import json

# 要保存的列表,网络预测的结果经过后处理的结果
data = [
    {
        "image_id": 0,
        "category_id": 62,
        "bbox": [80.71992,
                89.37024000000001,
                265.23024000000004,
                536.2201600000001],
        "score": 0.001
    },
    {
        "image_id": 1,
        "category_id": 2,
        "bbox": [115.45024,
                237.730246,
                26.06976,
                59.959804],
        "score": 0.001
    }
]

# 将列表保存到json文件
with open('data.json', 'w') as f:
    json.dump(data, f)


# 加载标注文件和结果文件，必须使用这两个函数来加载标签和结果
coco_gt = COCO('H:\\code\\datasets\\coco128\\annotations\\train.json')
coco_dt = coco_gt.loadRes('data.json')


# 初始化评估器
coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

# 运行评估
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# 获取评价指标
result = coco_eval.stats        # 变量含义与coco_eval.summarize()打印的顺序相同
print('AP50-95'+str(result[0]))
print('AP50'+str(result[0]))
print('AP75'+str(result[0]))