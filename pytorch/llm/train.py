# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   train.py
# @Time    :   2024/11/05 11:30:41
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   训练bert模型

import os
import sys
sys.path.append("./")
import torch
from torch import nn

from pytorch.llm.data import load_data_wiki
from pytorch.llm.bert import BERTModel
from pytorch.llm.utils import Timer, Accumulator, Animator


def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # 前向传播
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # 计算遮蔽语⾔模型损失
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # 计算下⼀句⼦预测任务的损失
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l


def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    # net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    net = net.to(devices)
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, Timer()
    animator = Animator(xlabel='step', ylabel='loss',
    xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # 遮蔽语⾔模型损失的和，下⼀句预测任务损失的和，句⼦对的数量，计数
    metric = Accumulator(4) 
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices)
            segments_X = segments_X.to(devices)
            valid_lens_x = valid_lens_x.to(devices)
            pred_positions_X = pred_positions_X.to(devices)
            mlm_weights_X = mlm_weights_X.to(devices)
            mlm_Y, nsp_y = mlm_Y.to(devices), nsp_y.to(devices)
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
            (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break
    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')


#-------------------------------------------------#
#   main函数
#-------------------------------------------------#
if __name__ == '__main__':
    batch_size, max_len = 512, 64
    train_iter, vocab = load_data_wiki(batch_size, max_len)


    net = BERTModel(len(vocab), num_hiddens=128, ffn_dim=256, num_heads=2,
                        num_layers=6, dropout=0.2, hid_in_features=128, 
                        mlm_in_features=128, nsp_in_features=128)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"   
    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss = nn.CrossEntropyLoss()
    train_bert(train_iter, net, loss, len(vocab), devices, 100)