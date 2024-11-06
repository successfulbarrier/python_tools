# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   bert-g.py
# @Time    :   2024/11/06 09:21:36
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   官方代码训练脚本

import sys
sys.path.append('./')
import torch
import torch.nn as nn
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertModel, BertForPreTraining
# from pytorch.llm.bert_g import BertModel
from pytorch.llm.utils import load_bert_base_weights


#-------------------------------------------------#
#   只加载主干的BERT模型，用于其他任务微调
#-------------------------------------------------#

class BertTrainer(nn.Module):
    def __init__(self, model_cfg_path, weight_path):
        super(BertTrainer, self).__init__()
        self.config = BertConfig.from_json_file(model_cfg_path)
        self.bert_model = BertModel(config=self.config, add_pooling_layer=False)  
        # 加载预训练权重,并冻结权重
        weights = load_bert_base_weights(weight_path)
        missing_keys, _ = self.bert_model.load_state_dict(weights, strict=False)
        # 打印未加载的参数
        if missing_keys:
            print("未加载的参数:")
            for key in missing_keys:
                print(key)   
        for param in self.bert_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        outputs = self.bert_model(x)
        return outputs                            
                                
#-------------------------------------------------#
#   main函数
#-------------------------------------------------#
if __name__ == '__main__':
    trainer = BertTrainer('/media/lht/D_Project/models/bert-tiny/config.json',
                          '/media/lht/D_Project/models/bert-tiny/pytorch_model.bin')
    # print(trainer)
    
