# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   train2.py
# @Time    :   2023/12/12 19:38:30
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   训练脚本

import sys
sys.path.append("J:\\code\\python_tools\\pytorch\\detection\\O_hand_fcos")

from model import FCOSDetector
import torch
from dataset import COCODataset
import math, time
from augment import Transforms
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel, SWALR  # 模型参数平均
from tools import ModelEMA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_gpu", type=str, default='0', help="number of cpu threads to use during batch generation")
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.n_gpu
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)

    # 加载数据集
    transform = Transforms()
    # train_dataset = COCODataset("H:\\code\\datasets\\coco128\\images\\train2017",
    #                             'H:\\code\\datasets\\coco128\\annotations\\train.json', resize_size=[640, 800], transform=transform)
    train_dataset = COCODataset("J:\\code\\datasets\\VOC_s\\train\\images",
                                'J:\\code\\datasets\\VOC_s\\train\\annotations\\train.json', resize_size=[640, 800], transform=transform)

    BATCH_SIZE = opt.batch_size
    EPOCHS = opt.epochs
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                            collate_fn=train_dataset.collate_fn,
                                            num_workers=opt.n_cpu, worker_init_fn=np.random.seed(0))
    
    # 定义模型
    model = FCOSDetector(mode="training").cuda()
    model = torch.nn.DataParallel(model)    # DP 多GPU训练
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("-->>模型参数量：" + str(params))   # 32159070

    # 优化器相关设置
    steps_per_epoch = len(train_dataset) // BATCH_SIZE
    TOTAL_STEPS = steps_per_epoch * EPOCHS
    WARMUP_STEPS = 500
    WARMUP_FACTOR = 1.0 / 3.0
    GLOBAL_STEPS = 0
    LR_INIT = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=LR_INIT, momentum=0.9, weight_decay=0.0001)

    def lr_func(step):
        lr = LR_INIT
        if step < WARMUP_STEPS:
            alpha = float(step) / WARMUP_STEPS
            warmup_factor = WARMUP_FACTOR * (1.0 - alpha) + alpha
            lr = lr * warmup_factor
        else:
            lr = 0.5*LR_INIT*(1+math.cos((step*math.pi)/TOTAL_STEPS))    # 余弦衰减策略
            if lr < 0.001:
                lr = 0.001
        return float(lr)
    
    # 模型参数平均swa类似于ema
    swa_start = 800   # 自动调整学习率起始轮数
    swa_scheduler = SWALR(optimizer, swa_lr=0.01)
    swa_model = AveragedModel(model)

    # ema
    ema = ModelEMA(model)

    # tensorboard
    writer = SummaryWriter()
    model.train()

    for epoch in range(EPOCHS):
        for epoch_step, (batch_imgs, batch_boxes, batch_classes) in enumerate(train_loader):

            # batch_imgs, batch_boxes, batch_classes = data
            batch_imgs = batch_imgs.cuda()
            batch_boxes = batch_boxes.cuda()
            batch_classes = batch_classes.cuda()

            start_time = time.time()

            optimizer.zero_grad()
            losses = model([batch_imgs, batch_boxes, batch_classes])
            loss = losses[-1]
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)    # 梯度裁剪，防止梯度爆炸
            optimizer.step()
            # swa
            # swa_model.update_parameters(model)  # 模型参数平均
            # 自定义调整学习率
            lr = lr_func(GLOBAL_STEPS)
            for param in optimizer.param_groups:
                param['lr'] = lr
            end_time = time.time()
            cost_time = int((end_time - start_time) * 1000)
            # ema
            ema.update(model)
            print(
                "global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f" % \
                (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, losses[0].mean(), losses[1].mean(),
                losses[2].mean(), cost_time, lr, loss.mean()))
            writer.add_scalar('train_loss', loss.mean(), GLOBAL_STEPS)
            writer.add_scalar('cls_loss', losses[0].mean(), GLOBAL_STEPS)
            writer.add_scalar('cnt_loss', losses[1].mean(), GLOBAL_STEPS)
            writer.add_scalar('reg_loss', losses[2].mean(), GLOBAL_STEPS)
            writer.add_scalar('lr', lr, GLOBAL_STEPS)
            GLOBAL_STEPS += 1

        torch.save(model.state_dict(), "./checkpoint/fcos/model_{}.pth".format(epoch + 1))


if __name__=="__main__":
    main()