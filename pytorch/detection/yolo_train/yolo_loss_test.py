# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   yolo_loss_load.py
# @Time    :   2023/11/14 22:12:25
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   新创建的python文件


# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   yolo_dataset_load.py
# @Time    :   2023/11/14 14:54:44
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   损失计算测试文件

import yolo_dataset_load.datasets as ds
import yaml
from yolo_model_load.yolo import Model
import yolo_loss_load.loss as Loss
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from yolo_dataset_load.general import labels_to_class_weights


'''	
    此文件功能:yolo格式的目标检测数据集加载
'''	


# main函数
if __name__ == '__main__':
    train_path = "H:\\code\\datasets\\coco128\\images\\train2017"
    hyp = "H:\\code\\python_tools\\pytorch\\detection\\yolo_train\\yolo_dataset_load\\hyp.scratch.yaml"
    cfg = "H:\\code\\python_tools\\pytorch\\detection\\yolo_train\\yolo_model_load\\yolov5l.yaml"
    dataset_yaml = "H:\\code\\python_tools\\pytorch\\detection\\yolo_train\\yolo_loss_load\\coco128.yaml"
    
    if isinstance(dataset_yaml, str):    
        with open(dataset_yaml, 'r', encoding='utf-8') as f:
            dataset_yaml = yaml.load(f.read(), Loader=yaml.FullLoader)
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict

    train_loader, dataset = ds.create_dataloader_train(train_path, 640, 2, 32, hyp=hyp, workers=4, shuffle=False, prefix="lht:")
    model = Model(cfg , ch=3, nc=80, anchors=hyp.get('anchors'))
    model.nc = 80  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, 80) * 80  # attach class weights
    model.names = dataset_yaml["names"]
    compute_loss = Loss.ComputeLoss(model)
    optimizer = Adam(model.parameters(), lr=hyp['lr0'], betas=(hyp['momentum'], 0.999)) 

    pbar = enumerate(train_loader)
    for i, (imgs, targets, paths, _) in pbar:
        print("------------"+str(i)+"-------------")
        print(imgs.shape, imgs.type)
        print(targets.shape, targets.type)
        imgs = imgs.float()/255
        optimizer.zero_grad()
        pred = model(imgs)
        print(pred[0].shape,pred[1].shape,pred[2].shape)
        loss, loss_items = compute_loss(pred, targets)
        print(str(loss_items[0]), str(loss_items[1]), str(loss_items[2]))
        loss.backward()
        optimizer.step()      

