# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   main.py
# @Time    :   2023/11/15 12:50:03
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   yolo算法测试主函数

# 官方库
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import optim
import math
from torch.cuda.amp import autocast, GradScaler
from collections import deque
import copy

# 自己实现的库
from cfg import Cfg
from utils import load_weight


# 三方库
import yolo_dataset_load.datasets as ds
from yolo_model_load.yolo import Model
import yolo_loss_load.loss as Loss
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from yolo_dataset_load.general import labels_to_class_weights
from yolo_dataset_load.torch_utils import select_device
from yolov8_model_load.yolov8 import Model_v8
from yolov8_loss_load.loss import v8DetectionLoss    
from yolo_val.val import val_my   


def v0():
    """
        最简单的训练流程只计算了损失，使用CPU
    """
    cfg = Cfg("test.yaml", "H:/code/python_tools/pytorch/detection/yolo_train/")
    train_loader, dataset = ds.create_dataloader_train(cfg.dataset_train, cfg.cfg_hyp["imgsz"], cfg.cfg_hyp["batch"], 
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht:")
    model = Model(cfg.cfg_model , ch=cfg.ch, nc=cfg.nc, anchors=cfg.cfg_hyp.get('anchors'))
    model.nc = cfg.nc  # attach number of classes to model
    model.hyp = cfg.cfg_hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, cfg.nc) * cfg.nc  # attach class weights
    model.names = cfg.names
    compute_loss = Loss.ComputeLoss(model)
    optimizer = Adam(model.parameters(), lr=cfg.cfg_hyp['lr0'], betas=(cfg.cfg_hyp['momentum'], 0.999)) 
    
    for epoch in range(cfg.epochs):
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


def v1():
    """
        最简单的训练流程只计算了损失，使用GPU
    """
    cfg = Cfg("test.yaml", "H:/code/python_tools/pytorch/detection/yolo_train/")  
    cfg.device = select_device(cfg.device, cfg.batch)
    train_loader, dataset = ds.create_dataloader_train(cfg.dataset_train, cfg.cfg_hyp["imgsz"], cfg.cfg_hyp["batch"], 
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht:")
    model = Model(cfg.cfg_model , ch=cfg.ch, nc=cfg.nc, anchors=cfg.cfg_hyp.get('anchors')).to(cfg.device)
    model.nc = cfg.nc  # attach number of classes to model
    model.hyp = cfg.cfg_hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, cfg.nc) * cfg.nc  # attach class weights
    model.names = cfg.names
    compute_loss = Loss.ComputeLoss(model)
    optimizer = Adam(model.parameters(), lr=cfg.cfg_hyp['lr0'], betas=(cfg.cfg_hyp['momentum'], 0.999)) 

    for epoch in range(cfg.epochs):
        pbar = enumerate(train_loader)
        for i, (imgs, targets, paths, _) in pbar:
            print("------------"+str(i)+"-------------")
            print(imgs.shape, imgs.type)
            print(targets.shape, targets.type)
            imgs = imgs.float().to(cfg.device)/255
            optimizer.zero_grad()
            pred = model(imgs)
            print(pred[0].shape,pred[1].shape,pred[2].shape)
            loss, loss_items = compute_loss(pred, targets.to(cfg.device))
            print(str(loss_items[0]), str(loss_items[1]), str(loss_items[2]))
            loss.backward()
            optimizer.step()   
    

def v2():
    """
        最简单的训练流程只计算了损失，使用GPU, yolov5模型  tensorboard --logdir runs --port=6007
    """
    cfg = Cfg("test.yaml", "H:/code/python_tools/pytorch/detection/yolo_train/")  
    cfg.device = select_device(cfg.device, cfg.batch)
    train_loader, dataset = ds.create_dataloader_train(cfg.dataset_train, cfg.cfg_hyp["imgsz"], cfg.cfg_hyp["batch"], 
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht:")
    model = Model(cfg.cfg_model , ch=cfg.ch, nc=cfg.nc, anchors=cfg.cfg_hyp.get('anchors')).to(cfg.device)
    load_weight(cfg.cfg_weight, model=model)
    model.nc = cfg.nc  # attach number of classes to model
    model.hyp = cfg.cfg_hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, cfg.nc) * cfg.nc  # attach class weights
    model.names = cfg.names
    compute_loss = Loss.ComputeLoss(model)
    optimizer = Adam(model.parameters(), lr=cfg.cfg_hyp['lr0'], betas=(cfg.cfg_hyp['momentum'], 0.999)) 
    writer = SummaryWriter()

    model.train()   # 进入训练模式
    for epoch in range(cfg.epochs):
        pbar = tqdm(enumerate(train_loader), desc="第"+str(epoch)+"轮:",total=train_loader.__len__())
        for i, (imgs, targets, paths, _) in pbar:
            n = i + epoch * train_loader.__len__()
            imgs = imgs.float().to(cfg.device)/255
            optimizer.zero_grad()
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets.to(cfg.device))
            writer.add_scalar('loss', loss, n)
            writer.add_scalar('loss box', loss_items[0].data, n)
            writer.add_scalar('loss obj', loss_items[1].data, n)
            writer.add_scalar('loss cls', loss_items[2].data, n) 
            loss.backward()
            optimizer.step() 

    writer.close()


def v3():
    """
        最简单的训练流程只计算了损失，使用GPU, yolov8模型
    """
    cfg = Cfg("test.yaml", "H:/code/python_tools/pytorch/detection/yolo_train/")  
    cfg.device = select_device(cfg.device, cfg.batch)
    train_loader, dataset = ds.create_dataloader_train(cfg.dataset_train, cfg.cfg_hyp["imgsz"], cfg.cfg_hyp["batch"], 
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht:")
    model = Model_v8(cfg.cfg_model , ch=cfg.ch, nc=cfg.nc).to(cfg.device)
    # 加载模型
    load_weight(cfg.cfg_weight, model)

    # 传递模型参数
    model.nc = cfg.nc  # attach number of classes to model
    model.hyp = cfg.cfg_hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, cfg.nc) * cfg.nc  # attach class weights
    model.names = cfg.names
    compute_loss = v8DetectionLoss(model)
    compute_loss.device = cfg.device
    optimizer = Adam(model.parameters(), lr=cfg.cfg_hyp['lr0'], betas=(cfg.cfg_hyp['momentum'], 0.999)) 
    writer = SummaryWriter()

    model.train()   # 进入训练模式
    for epoch in range(cfg.epochs):
        pbar = tqdm(enumerate(train_loader), desc="第"+str(epoch)+"轮:",total=train_loader.__len__())
        for i, (imgs, targets, paths, _) in pbar:
            n = i + epoch * train_loader.__len__()
            imgs = imgs.float().to(cfg.device)/255
            optimizer.zero_grad()
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets.to(cfg.device))
            # print(loss_items[0].data, loss_items[1].data, loss_items[2].data)
            writer.add_scalar('loss', loss, n)
            writer.add_scalar('loss box', loss_items[0].data, n)
            writer.add_scalar('loss cls', loss_items[1].data, n)
            writer.add_scalar('loss dfl', loss_items[2].data, n) 
            loss.backward()
            optimizer.step() 

    writer.close()


def v4():
    """
        最简单的训练流程只计算了损失，使用GPU, yolov5模型,加验证
    """
    cfg = Cfg("test.yaml", root_path="G:/code/python_tools/pytorch/detection/yolo_train/")
    cfg.device = select_device(cfg.device, cfg.batch)
    train_loader, dataset = ds.create_dataloader_train(cfg.dataset_train, cfg.cfg_hyp["imgsz"], 16,
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht:")
    val_loader, _ = ds.create_dataloader_train(cfg.dataset_val, cfg.cfg_hyp["imgsz"], cfg.cfg_hyp["batch"], 
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht:")
    
    model = Model(cfg.cfg_model , ch=cfg.ch, nc=cfg.nc, anchors=cfg.cfg_hyp.get('anchors')).to(cfg.device)
    # load_weight(cfg.cfg_weight, model=model)
    model.nc = cfg.nc  # attach number of classes to model
    model.hyp = cfg.cfg_hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, cfg.nc) * cfg.nc  # attach class weights
    model.names = cfg.names
    compute_loss = Loss.ComputeLoss(model)
    optimizer = Adam(model.parameters(), lr=cfg.cfg_hyp['lr0'], betas=(cfg.cfg_hyp['momentum'], 0.999)) 
    writer = SummaryWriter()

    # 进入训练模式
    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(enumerate(train_loader), desc="第"+str(epoch)+"轮:",total=train_loader.__len__())
        for i, (imgs, targets, paths, _) in pbar:
            n = i + epoch * train_loader.__len__()
            imgs = imgs.float().to(cfg.device)/255
            optimizer.zero_grad()
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets.to(cfg.device))
            writer.add_scalar('loss', loss, n)
            writer.add_scalar('loss box', loss_items[0].data, n)
            writer.add_scalar('loss obj', loss_items[1].data, n)
            writer.add_scalar('loss cls', loss_items[2].data, n) 
            loss.backward()
            optimizer.step() 
        run_my(val_loader, model, compute_loss, nc=cfg.nc)
    writer.close()    


def v5():
    """
        最简单的训练流程只计算了损失，使用GPU, yolov8模型,验证
    """
    cfg = Cfg("test.yaml", "H:/code/python_tools/pytorch/detection/yolo_train/")
    cfg.device = select_device(cfg.device, cfg.batch)
    train_loader, dataset = ds.create_dataloader_train(cfg.dataset_train, cfg.cfg_hyp["imgsz"], 16,
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht_train:")
    val_loader, _ = ds.create_dataloader_train(cfg.dataset_val, cfg.cfg_hyp["imgsz"], 8,
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht_val:")
    model = Model_v8(cfg.cfg_model , ch=cfg.ch, nc=cfg.nc).to(cfg.device)
    # 加载模型
    load_weight(cfg.cfg_weight, model)

    # 传递模型参数
    model.nc = cfg.nc  # attach number of classes to model
    model.hyp = cfg.cfg_hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, cfg.nc) * cfg.nc  # attach class weights
    model.names = cfg.names
    compute_loss = v8DetectionLoss(model)
    compute_loss.device = cfg.device
    optimizer = Adam(model.parameters(), lr=cfg.cfg_hyp['lr0'], betas=(cfg.cfg_hyp['momentum'], 0.999)) 
    writer = SummaryWriter()

    for epoch in range(cfg.epochs):
        model.train()  # 进入验证模式
        pbar = tqdm(enumerate(train_loader), desc="第"+str(epoch)+"轮:",total=train_loader.__len__())
        for i, (imgs, targets, paths, _) in pbar:
            n = i + epoch * train_loader.__len__()
            imgs = imgs.float().to(cfg.device)/255
            optimizer.zero_grad()
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets.to(cfg.device)) 
            loss.backward()
            optimizer.step()
            # tensorboard数据记录
            writer.add_scalar('loss', loss, n)
            writer.add_scalar('loss box', loss_items[0].data, n)
            writer.add_scalar('loss cls', loss_items[1].data, n)
            writer.add_scalar('loss dfl', loss_items[2].data, n)
        # 返回值说明：mp, mr, map50, map, loss box，loss cls，loss dfl
        result, _ = val_my(val_loader, model, compute_loss, nc=cfg.nc)       
        torch.save(model.state_dict(), 'yolov8_v5-1.pth')
    writer.close()


def v6():
    """
        最简单的训练流程只计算了损失，使用GPU, yolov8模型,验证
    """
    cfg = Cfg("test.yaml", "H:/code/python_tools/pytorch/detection/yolo_train/")
    cfg.device = select_device(cfg.device, cfg.batch)
    train_loader, dataset = ds.create_dataloader_train(cfg.dataset_train, cfg.cfg_hyp["imgsz"], 2,
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht_train:")
    val_loader, _ = ds.create_dataloader_train(cfg.dataset_val, cfg.cfg_hyp["imgsz"], 2,
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht_val:")
    model = Model_v8(cfg.cfg_model , ch=cfg.ch, nc=cfg.nc).to(cfg.device)
    # 加载模型
    load_weight(cfg.cfg_weight, model)

    # 传递模型参数
    model.nc = cfg.nc  # attach number of classes to model
    model.hyp = cfg.cfg_hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, cfg.nc) * cfg.nc  # attach class weights
    model.names = cfg.names
    compute_loss = v8DetectionLoss(model)
    compute_loss.device = cfg.device
    optimizer = Adam(model.parameters(), lr=cfg.cfg_hyp['lr0'], betas=(cfg.cfg_hyp['momentum'], 0.999)) 
    writer = SummaryWriter()

    for epoch in range(cfg.epochs):
        model.train()  # 进入验证模式
        pbar = tqdm(enumerate(train_loader), desc="第"+str(epoch)+"轮:",total=train_loader.__len__())
        for i, (imgs, targets, paths, _) in pbar:
            n = i + epoch * train_loader.__len__()
            imgs = imgs.float().to(cfg.device)/255
            optimizer.zero_grad()
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets.to(cfg.device)) 
            loss.backward()
            optimizer.step()
        # tensorboard数据记录
        writer.add_scalar('train_loss', loss, epoch)
        writer.add_scalar('train_lbox', loss_items[0].data, epoch)
        writer.add_scalar('train_lcls', loss_items[1].data, epoch)
        writer.add_scalar('train_ldfl', loss_items[2].data, epoch)
        # 返回值说明：mp, mr, map50, map, loss box，loss cls，loss dfl
        result, _ = val_my(val_loader, model, compute_loss, nc=cfg.nc)
        writer.add_scalar('mp', result[0], epoch) 
        writer.add_scalar('mr', result[1], epoch)
        writer.add_scalar('map50', result[2], epoch)
        writer.add_scalar('map50-95', result[3], epoch)
        writer.add_scalar('val_lbox', result[4], epoch)
        writer.add_scalar('val_lcls', result[5], epoch)
        writer.add_scalar('val_ldfl', result[6], epoch)        
        torch.save(model.state_dict(), 'yolov8_v5-2.pth')
    writer.close()


def v7():
    """
        最简单的训练流程只计算了损失，使用GPU, yolov8模型,验证
    """
    cfg = Cfg("test.yaml", "H:/code/python_tools/pytorch/detection/yolo_train/")
    cfg.device = select_device(cfg.device, cfg.batch)
    train_loader, dataset = ds.create_dataloader_train(cfg.dataset_train, cfg.cfg_hyp["imgsz"], 8,
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht_train:")
    val_loader, _ = ds.create_dataloader_train(cfg.dataset_val, cfg.cfg_hyp["imgsz"], 8,
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht_val:")
    model = Model_v8(cfg.cfg_model , ch=cfg.ch, nc=cfg.nc).to(cfg.device)
    # 加载模型
    load_weight(cfg.cfg_weight, model, train=False)

    # 传递模型参数
    model.nc = cfg.nc  # attach number of classes to model
    model.hyp = cfg.cfg_hyp  # attach hyperparameters to model
    # model.class_weights = labels_to_class_weights(dataset.labels, cfg.nc) * cfg.nc  # attach class weights
    model.names = cfg.names
    compute_loss = v8DetectionLoss(model)
    compute_loss.device = cfg.device
    optimizer = Adam(model.parameters(), lr=cfg.cfg_hyp['lr0'], betas=(cfg.cfg_hyp['momentum'], 0.999))
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.8)
    
    # 动态调整学习率
    start_epoch = 0     # 正常情况下为0
    # 以余弦方式衰减学习率
    # lf = lambda x: ((1 - math.cos(x * math.pi / cfg.epochs)) / 2) * (cfg.lrf - 1) + 1
    lf = lambda x: (1 - x / cfg.epochs) * (1.0 - cfg.lrf) + cfg.lrf
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1 
    writer = SummaryWriter()

    for epoch in range(cfg.epochs):
        model.train()  # 进入验证模式
        pbar = tqdm(enumerate(train_loader), desc="第"+str(epoch)+"轮:",total=train_loader.__len__())
        for i, (imgs, targets, paths, _) in pbar:
            n = i + epoch * train_loader.__len__()
            imgs = imgs.float().to(cfg.device)/255
            optimizer.zero_grad()
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets.to(cfg.device)) 
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()    # 释放
            writer.add_scalar('train_loss', loss, n)
            writer.add_scalar('train_lbox', loss_items[0].data, n)
            writer.add_scalar('train_lcls', loss_items[1].data, n)
            writer.add_scalar('train_ldfl', loss_items[2].data, n)
        # tensorboard数据记录
        writer.add_scalar('train_loss', loss, epoch)
        writer.add_scalar('train_lbox', loss_items[0].data, epoch)
        writer.add_scalar('train_lcls', loss_items[1].data, epoch)
        writer.add_scalar('train_ldfl', loss_items[2].data, epoch)
        # 返回值说明：mp, mr, map50, map, loss box，loss cls，loss dfl
        result, _ = val_my(val_loader, model, compute_loss, nc=cfg.nc)
        writer.add_scalar('mp', result[0], epoch) 
        writer.add_scalar('mr', result[1], epoch)
        writer.add_scalar('map50', result[2], epoch)
        writer.add_scalar('map50-95', result[3], epoch)
        writer.add_scalar('val_lbox', result[4], epoch)
        writer.add_scalar('val_lcls', result[5], epoch)
        writer.add_scalar('val_ldfl', result[6], epoch)        
        torch.save(model.state_dict(), 'yolov8_v5-2.pth')
        # 学习率调整
        scheduler.step()
        # print('学习率:'+str(scheduler.get_last_lr()))
    writer.close()


def v8():
    """
        最简单的训练流程只计算了损失，使用GPU, yolov8模型,验证，混合精度训练
    """
    cfg = Cfg("test.yaml", "H:/code/python_tools/pytorch/detection/yolo_train/")
    cfg.device = select_device(cfg.device, cfg.batch)
    train_loader, dataset = ds.create_dataloader_train(cfg.dataset_train, cfg.cfg_hyp["imgsz"], 12,
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht_train:")
    val_loader, _ = ds.create_dataloader_train(cfg.dataset_val, cfg.cfg_hyp["imgsz"], 12,
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht_val:")
    model = Model_v8(cfg.cfg_model , ch=cfg.ch, nc=cfg.nc).to(cfg.device)
    # 加载模型
    load_weight(cfg.cfg_weight, model, train=False)

    # 传递模型参数
    model.nc = cfg.nc  # attach number of classes to model
    model.hyp = cfg.cfg_hyp  # attach hyperparameters to model
    # model.class_weights = labels_to_class_weights(dataset.labels, cfg.nc) * cfg.nc  # attach class weights
    model.names = cfg.names
    compute_loss = v8DetectionLoss(model)
    compute_loss.device = cfg.device
    optimizer = Adam(model.parameters(), lr=cfg.cfg_hyp['lr0'], betas=(cfg.cfg_hyp['momentum'], 0.999))
    # optimizer = torch.optim.SGD(model.parameters(), lr=cfg.cfg_hyp['lr0'], momentum=0.8)
    # 动态调整学习率
    start_epoch = 0     # 正常情况下为
    # 以余弦方式衰减学习率
    # lf = lambda x: ((1 - math.cos(x * math.pi / cfg.epochs)) / 2) * (cfg.lrf - 1) + 1
    lf = lambda x: (1 - x / cfg.epochs) * (1.0 - cfg.lrf) + cfg.lrf
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1 
    writer = SummaryWriter()

    # 创建混合精度训练的GradScaler
    scaler = GradScaler()       

    for epoch in range(cfg.epochs):
        model.train()  # 进入验证模式
        pbar = tqdm(enumerate(train_loader), desc="第"+str(epoch)+"轮:",total=train_loader.__len__())
        for i, (imgs, targets, paths, _) in pbar:
            n = i + epoch * train_loader.__len__()
            imgs = imgs.float().to(cfg.device)/255
            optimizer.zero_grad()
            with autocast():    # 使用混合精度训练
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(cfg.device)) 
            # 使用混合精度反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.empty_cache()    # 释放无用的GPU内存
            # tensorboard数据记录
            writer.add_scalar('train_loss', loss, n)
            writer.add_scalar('train_lbox', loss_items[0].data, n)
            writer.add_scalar('train_lcls', loss_items[1].data, n)
            writer.add_scalar('train_ldfl', loss_items[2].data, n)
        # 返回值说明：mp, mr, map50, map, loss box，loss cls，loss dfl
        result, _ = val_my(val_loader, model, compute_loss, nc=cfg.nc)
        writer.add_scalar('mp', result[0], epoch) 
        writer.add_scalar('mr', result[1], epoch)
        writer.add_scalar('map50', result[2], epoch)
        writer.add_scalar('map50-95', result[3], epoch)
        writer.add_scalar('val_lbox', result[4], epoch)
        writer.add_scalar('val_lcls', result[5], epoch)
        writer.add_scalar('val_ldfl', result[6], epoch)        
        torch.save(model.state_dict(), 'yolov8_v5-2.pth')
        # 学习率调整
        scheduler.step()
        # print('学习率:'+str(scheduler.get_last_lr()))
    writer.close()


def v9():
    """
        最简单的训练流程只计算了损失，使用GPU, yolov8模型,验证，模型平均
    """
    cfg = Cfg("test.yaml", "J:/code/python_tools/pytorch/detection/yolo_train/")
    cfg.device = select_device(cfg.device, cfg.batch)
    train_loader, dataset = ds.create_dataloader_train(cfg.dataset_train, cfg.cfg_hyp["imgsz"], 16,
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht_train:")
    val_loader, _ = ds.create_dataloader_train(cfg.dataset_val, cfg.cfg_hyp["imgsz"], 16,
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht_val:")
    model = Model_v8(cfg.cfg_model , ch=cfg.ch, nc=cfg.nc).to(cfg.device)
    # 加载模型
    load_weight(cfg.cfg_weight, model)

    # 传递模型参数
    model.nc = cfg.nc  # attach number of classes to model
    model.hyp = cfg.cfg_hyp  # attach hyperparameters to model
    # model.class_weights = labels_to_class_weights(dataset.labels, cfg.nc) * cfg.nc  # attach class weights
    model.names = cfg.names
    compute_loss = v8DetectionLoss(model)
    compute_loss.device = cfg.device
    optimizer = Adam(model.parameters(), lr=cfg.cfg_hyp['lr0'], betas=(cfg.cfg_hyp['momentum'], 0.999))

    # 动态调整学习率
    start_epoch = 0     # 正常情况下为0
    # 以余弦方式衰减学习率
    # lf = lambda x: ((1 - math.cos(x * math.pi / cfg.epochs)) / 2) * (cfg.lrf - 1) + 1
    lf = lambda x: (1 - x / cfg.epochs) * (1.0 - cfg.lrf) + cfg.lrf
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1 
    writer = SummaryWriter()

    # 创建用于保存的队列
    queue = deque(maxlen=4)

    for epoch in range(cfg.epochs):
        model.train()  # 进入验证模式
        pbar = tqdm(enumerate(train_loader), desc="第"+str(epoch)+"轮:",total=train_loader.__len__())
        
        # 深拷贝模型并压入队列
        model_copy = copy.deepcopy(model)
        queue.append(model_copy)

        # 模型参数平均
        if len(queue) == 4:
            with torch.no_grad():
                for param1, param2, param3, param4 in zip(queue[0].parameters(), queue[1].parameters(), queue[2].parameters(), queue[3].parameters()):
                    param1.data.add_(param2.data).add_(param3.data).add_(param4.data).mul_(1/4)

        for i, (imgs, targets, paths, _) in pbar:
            n = i + epoch * train_loader.__len__()
            imgs = imgs.float().to(cfg.device)/255
            optimizer.zero_grad()
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets.to(cfg.device)) 
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()    # 释放
        # tensorboard数据记录
        writer.add_scalar('train_loss', loss, epoch)
        writer.add_scalar('train_lbox', loss_items[0].data, epoch)
        writer.add_scalar('train_lcls', loss_items[1].data, epoch)
        writer.add_scalar('train_ldfl', loss_items[2].data, epoch)
        # 返回值说明：mp, mr, map50, map, loss box，loss cls，loss dfl
        result, _ = val_my(val_loader, model, compute_loss, nc=cfg.nc)
        writer.add_scalar('mp', result[0], epoch) 
        writer.add_scalar('mr', result[1], epoch)
        writer.add_scalar('map50', result[2], epoch)
        writer.add_scalar('map50-95', result[3], epoch)
        writer.add_scalar('val_lbox', result[4], epoch)
        writer.add_scalar('val_lcls', result[5], epoch)
        writer.add_scalar('val_ldfl', result[6], epoch)        
        torch.save(model.state_dict(), 'yolov8_v5-2.pth')
        # 学习率调整
        scheduler.step()
        # print('学习率:'+str(scheduler.get_last_lr()))
    writer.close()


def v10():
    """
        最简单的训练流程只计算了损失，使用GPU, yolov8模型,验证，混合精度训练和模型参数平均
    """
    cfg = Cfg("test.yaml", "H:/code/python_tools/pytorch/detection/yolo_train/")
    cfg.device = select_device(cfg.device, cfg.batch)
    train_loader, dataset = ds.create_dataloader_train(cfg.dataset_train, cfg.cfg_hyp["imgsz"], cfg.cfg_hyp["batch"],
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht_train:")
    val_loader, _ = ds.create_dataloader_train(cfg.dataset_val, cfg.cfg_hyp["imgsz"], cfg.cfg_hyp["batch"],
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht_val:")
    model = Model_v8(cfg.cfg_model , ch=cfg.ch, nc=cfg.nc).to(cfg.device)
    # 加载模型
    load_weight(cfg.cfg_weight, model,train=False)

    # 传递模型参数
    model.nc = cfg.nc  # attach number of classes to model
    model.hyp = cfg.cfg_hyp  # attach hyperparameters to model
    # model.class_weights = labels_to_class_weights(dataset.labels, cfg.nc) * cfg.nc  # attach class weights
    model.names = cfg.names
    compute_loss = v8DetectionLoss(model)
    compute_loss.device = cfg.device
    optimizer = Adam(model.parameters(), lr=cfg.cfg_hyp['lr0'], betas=(cfg.cfg_hyp['momentum'], 0.999))
    # optimizer = SGD(model.parameters(), lr=cfg.cfg_hyp['lr0'], momentum=0.8)
    # 动态调整学习率
    start_epoch = 0     # 正常情况下为0
    # 以余弦方式衰减学习率
    # lf = lambda x: ((1 - math.cos(x * math.pi / cfg.epochs)) / 2) * (cfg.lrf - 1) + 1
    lf = lambda x: (1 - x / cfg.epochs) * (1.0 - cfg.lrf) + cfg.lrf
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1 
    writer = SummaryWriter()

    # 创建混合精度训练的GradScaler
    scaler = GradScaler()       

    # 创建用于保存的队列
    queue = deque(maxlen=4)

    for epoch in range(cfg.epochs):
        model.train()  # 进入训练模式
        pbar = tqdm(enumerate(train_loader), desc="第"+str(epoch)+"轮:",total=train_loader.__len__())
        
        # 深拷贝模型并压入队列
        model_copy = copy.deepcopy(model)
        queue.append(model_copy)

        # 模型参数平均
        if len(queue) == 4:
            with torch.no_grad():
                for modelz, param1, param2, param3, param4 in zip(model.parameters(), queue[0].parameters(), queue[1].parameters(), queue[2].parameters(), queue[3].parameters()):
                    modelz.data = param1.data.add_(param2.data).add_(param3.data).add_(param4.data).mul_(1/4)
            
        for i, (imgs, targets, paths, _) in pbar:
            n = i + epoch * train_loader.__len__()
            imgs = imgs.float().to(cfg.device)/255
            optimizer.zero_grad()
            with autocast():    # 使用混合精度训练
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(cfg.device)) 
            # 使用混合精度反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.empty_cache()    # 释放无用的GPU内存
        # tensorboard数据记录
        writer.add_scalar('train_loss', loss, epoch)
        writer.add_scalar('train_lbox', loss_items[0].data, epoch)
        writer.add_scalar('train_lcls', loss_items[1].data, epoch)
        writer.add_scalar('train_ldfl', loss_items[2].data, epoch)
        # 返回值说明：mp, mr, map50, map, loss box，loss cls，loss dfl
        result, _ = val_my(val_loader, model, compute_loss, nc=cfg.nc)
        writer.add_scalar('mp', result[0], epoch) 
        writer.add_scalar('mr', result[1], epoch)
        writer.add_scalar('map50', result[2], epoch)
        writer.add_scalar('map50-95', result[3], epoch)
        writer.add_scalar('val_lbox', result[4], epoch)
        writer.add_scalar('val_lcls', result[5], epoch)
        writer.add_scalar('val_ldfl', result[6], epoch)        
        torch.save(model.state_dict(), 'yolov8_v5-2.pth')
        # 学习率调整
        scheduler.step()
        # print('学习率:'+str(scheduler.get_last_lr()))
    writer.close()


def v11():
    """
        最简单的训练流程只计算了损失，使用GPU, yolov8模型,验证，使用了梯度截断，可以正常的进行迁移学习
    """
    cfg = Cfg("test.yaml", "H:/code/python_tools/pytorch/detection/yolo_train/")
    cfg.device = select_device(cfg.device, cfg.batch)
    train_loader, dataset = ds.create_dataloader_train(cfg.dataset_train, cfg.cfg_hyp["imgsz"], cfg.cfg_hyp["batch"],
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht_train:")
    val_loader, _ = ds.create_dataloader_train(cfg.dataset_val, cfg.cfg_hyp["imgsz"], 8,
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht_val:")
    model = Model_v8(cfg.cfg_model , ch=cfg.ch, nc=cfg.nc).to(cfg.device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("-->>模型参数量：" + str(params))   # 3014732
    
    # 加载模型
    load_weight(cfg.cfg_weight, model)
    # 打包模型方便多GUP训练
    # model = torch.nn.DataParallel(model)
    
    # 传递模型参数
    model.nc = cfg.nc  # attach number of classes to model
    model.hyp = cfg.cfg_hyp  # attach hyperparameters to model
    # model.class_weights = labels_to_class_weights(dataset.labels, cfg.nc) * cfg.nc  # attach class weights
    model.names = cfg.names
    compute_loss = v8DetectionLoss(model)
    compute_loss.device = cfg.device

    # 动态调整学习率
    per_epoch = train_loader.__len__()
    total_steps = per_epoch * cfg.epochs
    # 学习率热身策略
    warmup_steps = 500
    # 定义优化器
    global_steps = 0
    init_lr = cfg.cfg_hyp['lr0']
    # 调整momentum和weight_decay的值在在一定程度哦上可以起到防止过拟合的作用
    # momentum越小作用越强，weight_decay越大作用越强
    # optimizer = Adam(model.parameters(), lr=cfg.cfg_hyp['lr0'], betas=(cfg.cfg_hyp['momentum'], 0.999))
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=0.0001)

    # 学习率调整函数
    def lr_func(step):
        lr = init_lr
        warmup_factor = 1.0/3.0
        if step < warmup_steps:
            alpha = float(step) / warmup_steps
            warmup_factor = warmup_factor * (1.0 - alpha) + alpha
            lr = lr * warmup_factor
        else:
            lr = 0.5*lr*(1+math.cos((step*math.pi)/total_steps))    # 余弦衰减策略
            if lr < 0.001:
                lr = 0.001
        return float(lr)

    writer = SummaryWriter()

    for epoch in range(cfg.epochs):
        model.train()  # 进入验证模式
        pbar = tqdm(enumerate(train_loader), desc="第"+str(epoch)+"轮:", total=train_loader.__len__())
        for i, (imgs, targets, paths, _) in pbar:
            imgs = imgs.float().to(cfg.device)/255
            
            # 调整学习率
            lr = lr_func(global_steps)
            for param in optimizer.param_groups:
                param['lr'] = lr

            optimizer.zero_grad()
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets.to(cfg.device)) 
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)    # 梯度裁剪，防止梯度爆炸调整合适可以加速模型收敛
            optimizer.step()
            
            torch.cuda.empty_cache()    # 释放
            writer.add_scalar('train_loss', loss, global_steps)
            writer.add_scalar('train_lbox', loss_items[0].data, global_steps)
            writer.add_scalar('train_lcls', loss_items[1].data, global_steps)
            writer.add_scalar('train_ldfl', loss_items[2].data, global_steps)
            writer.add_scalar('lr', lr, global_steps)
            global_steps += 1

        # # tensorboard数据记录
        # writer.add_scalar('train_loss', loss, epoch)
        # writer.add_scalar('train_lbox', loss_items[0].data, epoch)
        # writer.add_scalar('train_lcls', loss_items[1].data, epoch)
        # writer.add_scalar('train_ldfl', loss_items[2].data, epoch)
        # 返回值说明：mp, mr, map50, map, loss box，loss cls，loss dfl
        result, _ = val_my(val_loader, model, compute_loss, nc=cfg.nc)
        writer.add_scalar('mp', result[0], epoch) 
        writer.add_scalar('mr', result[1], epoch)
        writer.add_scalar('map50', result[2], epoch)
        writer.add_scalar('map50-95', result[3], epoch)
        writer.add_scalar('val_lbox', result[4], epoch)
        writer.add_scalar('val_lcls', result[5], epoch)
        writer.add_scalar('val_ldfl', result[6], epoch)        
        torch.save(model.state_dict(), 'yolov8_v5-2.pth')
        # 学习率调整
        
        # print('学习率:'+str(scheduler.get_last_lr()))
    writer.close()


def v12():
    """
        最简单的训练流程只计算了损失，使用GPU, yolov8模型,验证，使用了梯度截断，可以正常的进行迁移学习，混合精度训练
    """
    cfg = Cfg("test.yaml", "H:/code/python_tools/pytorch/detection/yolo_train/")
    cfg.device = select_device(cfg.device, cfg.batch)
    train_loader, dataset = ds.create_dataloader_train(cfg.dataset_train, cfg.cfg_hyp["imgsz"], cfg.cfg_hyp["batch"],
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht_train:")
    val_loader, _ = ds.create_dataloader_train(cfg.dataset_val, cfg.cfg_hyp["imgsz"], 8,
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht_val:")
    model = Model_v8(cfg.cfg_model , ch=cfg.ch, nc=cfg.nc).to(cfg.device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("-->>模型参数量：" + str(params))   # 3014732
    
    # 加载模型
    load_weight(cfg.cfg_weight, model)
    # 打包模型方便多GUP训练
    # model = torch.nn.DataParallel(model)
    
    # 传递模型参数
    model.nc = cfg.nc  # attach number of classes to model
    model.hyp = cfg.cfg_hyp  # attach hyperparameters to model
    # model.class_weights = labels_to_class_weights(dataset.labels, cfg.nc) * cfg.nc  # attach class weights
    model.names = cfg.names
    compute_loss = v8DetectionLoss(model)
    compute_loss.device = cfg.device

    # 动态调整学习率
    per_epoch = train_loader.__len__()
    total_steps = per_epoch * cfg.epochs
    # 学习率热身策略
    warmup_steps = 500
    # 定义优化器
    global_steps = 0
    init_lr = cfg.cfg_hyp['lr0']
    # 调整momentum和weight_decay的值在在一定程度哦上可以起到防止过拟合的作用
    # momentum越小作用越强，weight_decay越大作用越强
    # optimizer = Adam(model.parameters(), lr=cfg.cfg_hyp['lr0'], betas=(cfg.cfg_hyp['momentum'], 0.999))
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=0.0001)

    # 学习率调整函数
    def lr_func(step):
        lr = init_lr
        warmup_factor = 1.0/3.0
        if step < warmup_steps:
            alpha = float(step) / warmup_steps
            warmup_factor = warmup_factor * (1.0 - alpha) + alpha
            lr = lr * warmup_factor
        else:
            lr = 0.5*lr*(1+math.cos((step*math.pi)/total_steps))    # 余弦衰减策略
            if lr < 0.001:
                lr = 0.001
        return float(lr)

    writer = SummaryWriter()

    # 创建混合精度训练的GradScaler
    scaler = GradScaler()

    for epoch in range(cfg.epochs):
        model.train()  # 进入验证模式
        pbar = tqdm(enumerate(train_loader), desc="第"+str(epoch)+"轮:", total=train_loader.__len__())
        for i, (imgs, targets, paths, _) in pbar:
            imgs = imgs.float().to(cfg.device)/255
            
            # 调整学习率
            lr = lr_func(global_steps)
            for param in optimizer.param_groups:
                param['lr'] = lr

            optimizer.zero_grad()
            with autocast():    # 使用混合精度训练
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(cfg.device)) 
            # 使用混合精度反向传播
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)    # 梯度裁剪，防止梯度爆炸调整合适可以加速模型收敛
            scaler.step(optimizer)
            scaler.update()
            
            torch.cuda.empty_cache()    # 释放
            writer.add_scalar('train_loss', loss, global_steps)
            writer.add_scalar('train_lbox', loss_items[0].data, global_steps)
            writer.add_scalar('train_lcls', loss_items[1].data, global_steps)
            writer.add_scalar('train_ldfl', loss_items[2].data, global_steps)
            writer.add_scalar('lr', lr, global_steps)
            global_steps += 1

        # # tensorboard数据记录
        # writer.add_scalar('train_loss', loss, epoch)
        # writer.add_scalar('train_lbox', loss_items[0].data, epoch)
        # writer.add_scalar('train_lcls', loss_items[1].data, epoch)
        # writer.add_scalar('train_ldfl', loss_items[2].data, epoch)
        # 返回值说明：mp, mr, map50, map, loss box，loss cls，loss dfl
        result, _ = val_my(val_loader, model, compute_loss, nc=cfg.nc)
        writer.add_scalar('mp', result[0], epoch) 
        writer.add_scalar('mr', result[1], epoch)
        writer.add_scalar('map50', result[2], epoch)
        writer.add_scalar('map50-95', result[3], epoch)
        writer.add_scalar('val_lbox', result[4], epoch)
        writer.add_scalar('val_lcls', result[5], epoch)
        writer.add_scalar('val_ldfl', result[6], epoch)        
        torch.save(model.state_dict(), 'yolov8_v5-2.pth')
        # 学习率调整
        
        # print('学习率:'+str(scheduler.get_last_lr()))
    writer.close()


def v13():
    """
        最简单的训练流程只计算了损失，使用GPU, yolov8模型,验证，使用了梯度截断，可以正常的进行迁移学习，混合精度训练，模型参数平均
    """
    cfg = Cfg("test.yaml", "H:/code/python_tools/pytorch/detection/yolo_train/")
    cfg.device = select_device(cfg.device, cfg.batch)
    train_loader, dataset = ds.create_dataloader_train(cfg.dataset_train, cfg.cfg_hyp["imgsz"], cfg.cfg_hyp["batch"],
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht_train:")
    val_loader, _ = ds.create_dataloader_train(cfg.dataset_val, cfg.cfg_hyp["imgsz"], 8,
                                                       32, hyp=cfg.cfg_hyp, workers=cfg.cfg_hyp["workers"], 
                                                       shuffle=False, prefix="lht_val:")
    model = Model_v8(cfg.cfg_model , ch=cfg.ch, nc=cfg.nc).to(cfg.device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("-->>模型参数量：" + str(params))   # 3014732
    
    # 加载模型
    load_weight(cfg.cfg_weight, model)
    # 打包模型方便多GUP训练
    # model = torch.nn.DataParallel(model)
    
    # 传递模型参数
    model.nc = cfg.nc  # attach number of classes to model
    model.hyp = cfg.cfg_hyp  # attach hyperparameters to model
    # model.class_weights = labels_to_class_weights(dataset.labels, cfg.nc) * cfg.nc  # attach class weights
    model.names = cfg.names
    compute_loss = v8DetectionLoss(model)
    compute_loss.device = cfg.device

    # 动态调整学习率
    per_epoch = train_loader.__len__()
    total_steps = per_epoch * cfg.epochs
    # 学习率热身策略
    warmup_steps = 500
    # 定义优化器
    global_steps = 0
    init_lr = cfg.cfg_hyp['lr0']
    # 调整momentum和weight_decay的值在在一定程度哦上可以起到防止过拟合的作用
    # momentum越小作用越强，weight_decay越大作用越强
    # optimizer = Adam(model.parameters(), lr=cfg.cfg_hyp['lr0'], betas=(cfg.cfg_hyp['momentum'], 0.999))
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=0.0001)

    # 学习率调整函数
    def lr_func(step):
        lr = init_lr
        warmup_factor = 1.0/3.0
        if step < warmup_steps:
            alpha = float(step) / warmup_steps
            warmup_factor = warmup_factor * (1.0 - alpha) + alpha
            lr = lr * warmup_factor
        else:
            lr = 0.5*lr*(1+math.cos((step*math.pi)/total_steps))    # 余弦衰减策略
            if lr < 0.001:
                lr = 0.001
        return float(lr)

    writer = SummaryWriter()

    # 创建混合精度训练的GradScaler
    scaler = GradScaler()

    # 创建用于保存的队列
    queue = deque(maxlen=4)

    for epoch in range(cfg.epochs):
        model.train()  # 进入验证模式
        pbar = tqdm(enumerate(train_loader), desc="第"+str(epoch)+"轮:", total=train_loader.__len__())
        
        # 深拷贝模型并压入队列
        model_copy = copy.deepcopy(model)
        queue.append(model_copy)

        # 模型参数平均
        if len(queue) == 4:
            with torch.no_grad():
                for modelz, param1, param2, param3, param4 in zip(model.parameters(), queue[0].parameters(), queue[1].parameters(), queue[2].parameters(), queue[3].parameters()):
                    modelz.data = param1.data.add_(param2.data).add_(param3.data).add_(param4.data).mul_(1/4)

        for i, (imgs, targets, paths, _) in pbar:
            imgs = imgs.float().to(cfg.device)/255
            
            # 调整学习率
            lr = lr_func(global_steps)
            for param in optimizer.param_groups:
                param['lr'] = lr

            optimizer.zero_grad()
            with autocast():    # 使用混合精度训练
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(cfg.device)) 
            # 使用混合精度反向传播
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)    # 梯度裁剪，防止梯度爆炸调整合适可以加速模型收敛
            scaler.step(optimizer)
            scaler.update()
            
            torch.cuda.empty_cache()    # 释放
            writer.add_scalar('train_loss', loss, global_steps)
            writer.add_scalar('train_lbox', loss_items[0].data, global_steps)
            writer.add_scalar('train_lcls', loss_items[1].data, global_steps)
            writer.add_scalar('train_ldfl', loss_items[2].data, global_steps)
            writer.add_scalar('lr', lr, global_steps)
            global_steps += 1

        # # tensorboard数据记录
        # writer.add_scalar('train_loss', loss, epoch)
        # writer.add_scalar('train_lbox', loss_items[0].data, epoch)
        # writer.add_scalar('train_lcls', loss_items[1].data, epoch)
        # writer.add_scalar('train_ldfl', loss_items[2].data, epoch)
        # 返回值说明：mp, mr, map50, map, loss box，loss cls，loss dfl
        result, _ = val_my(val_loader, model, compute_loss, nc=cfg.nc)
        writer.add_scalar('mp', result[0], epoch) 
        writer.add_scalar('mr', result[1], epoch)
        writer.add_scalar('map50', result[2], epoch)
        writer.add_scalar('map50-95', result[3], epoch)
        writer.add_scalar('val_lbox', result[4], epoch)
        writer.add_scalar('val_lcls', result[5], epoch)
        writer.add_scalar('val_ldfl', result[6], epoch)        
        torch.save(model.state_dict(), 'yolov8_v5-2.pth')
        # 学习率调整
        
        # print('学习率:'+str(scheduler.get_last_lr()))
    writer.close()



# main函数
if __name__ == '__main__':
    v13()



