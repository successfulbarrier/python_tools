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
from transformers.models.bert.modeling_bert import BertForPreTraining
from transformers.models.bert.configuration_bert import BertConfig
from transformers import BertTokenizer

from pytorch.llm.utils import set_seed


#-------------------------------------------------#
#   main函数
#-------------------------------------------------#
if __name__ == '__main__':
    # 设置随机种子
    set_seed(1)
    # 加载数据集
    tokenizer = BertTokenizer.from_pretrained('/media/lht/D_Project/models/bert-tiny')
    # 加载模型
    config = BertConfig.from_json_file("/media/lht/D_Project/models/bert-tiny/config.json")
    bert_model = BertForPreTraining(config=config)  
    # 加载预训练权重,并冻结权重
    weights = torch.load("/media/lht/D_Project/models/bert-tiny/pytorch_model.bin")
    missing_keys, _ = bert_model.load_state_dict(weights, strict=False)
    # 打印未加载的参数
    if missing_keys:
        print("未加载的参数:")
        for key in missing_keys:
            print(key) 
    # 设置训练设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"   
    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 测试数据
    text_tesnsor = tokenizer(["I am [MASK] person", 
                              "I am [MASK] person", 
                              "I am [MASK] person"], return_tensors="pt")
    print(text_tesnsor)
    
    bert_model = bert_model.to(devices)
    trainer = torch.optim.Adam(bert_model.parameters(), lr=0.001)
    for i in range(10):
        print(i)
        #-------------------------------------------------#
        #   input_ids: 输入的文本
        #   attention_mask: 输入的掩码
        #   labels: 标签，-100表示忽略的部分，[mask]位置用真实的标签
        #   next_sentence_label: 是否是连续的两个句子，0表示是连续，1表示是随机的
        #-------------------------------------------------#
        loss = bert_model(input_ids=text_tesnsor["input_ids"].to(devices), attention_mask=text_tesnsor["attention_mask"].to(devices),
                        labels=torch.tensor([[-100, -100, -100, 1037, -100, -100],[-100, -100, -100, 1037, -100, -100],[-100, -100, -100, 1037, -100, -100]]).to(devices), 
                        next_sentence_label=torch.tensor([0, 0, 0]).to(devices), return_dict=False)
        print(loss[0])
        loss[0].backward()
        trainer.step()
        