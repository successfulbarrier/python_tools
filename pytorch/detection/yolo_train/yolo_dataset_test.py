# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   yolo_dataset_load.py
# @Time    :   2023/11/14 14:54:44
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   数据加载测试文件

import yolo_dataset_load.datasets as ds
import yaml


# main函数
if __name__ == '__main__':
    train_path = "H:\\code\\datasets\\coco128\\images\\train2017"
    hyp = "H:\\code\\python_tools\\pytorch\\detection\\yolo_train\\yolo_dataset_load\\hyp.scratch.yaml"

    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict

    train_loader, dataset = ds.create_dataloader_train(train_path, 640, 2, 32, hyp=hyp, workers=4, shuffle=False, prefix="lht:")

    pbar = enumerate(train_loader)
    for i, (imgs, targets, paths, _) in pbar:
        print("------------"+str(i)+"-------------")
        print(imgs.shape, imgs.type)
        print(targets.shape, targets.type)
     

