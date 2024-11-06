# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   bert.py
# @Time    :   2024/11/03 21:26:21
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   新创建的python脚本

import torch
import torch.nn as nn


class BERTEncoder(nn.Module):
    """
    BERTEncoder 类的构造函数。

    Args:
        vocab_size (int): 词汇表的大小。
        num_hiddens (int): 隐藏层的大小。
        ffn_dim (int): 前馈网络的维度。
        num_heads (int): 多头注意力机制中的头数。
        num_layers (int): Transformer 编码器的层数。
        dropout (float): Dropout 的比率。
        max_len (int, optional): 输入序列的最大长度。默认为 1000。
        Encoder_dim (int, optional): 编码器的维度。默认为 768。
        **kwargs: 其他关键字参数。

    Returns:
        None

    """
    def __init__(self, vocab_size, num_hiddens, ffn_dim,
                num_heads, num_layers, dropout,
                max_len=1000, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                    num_hiddens, num_heads, ffn_dim, dropout, 
                    batch_first=True,bias=True), num_layers)
        # 在BERT中，位置嵌⼊是可学习的，因此我们创建⼀个⾜够⻓的位置嵌⼊参数
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))
        
    def forward(self, tokens, segments, valid_lens):
        """
        前向传播函数。

        Args:
            tokens (Tensor): 输入的token IDs，形状为 (批量大小, 最大序列长度)。
            segments (Tensor): 输入的segment IDs，形状为 (批量大小, 最大序列长度)。
            valid_lens (Tensor): 每个样本的有效长度，形状为 (批量大小,)。

        Returns:
            Tensor: 经过多层Transformer块处理后的输出，形状为 (批量大小, 最大序列长度, num_hiddens)。

        """
        # 将有效位数设置为True，其余位置设置为False
        valid_mask = torch.zeros((valid_lens.size(0), tokens.shape[1]), dtype=torch.bool)  # 初始化全为0的二维 Tensor
        for i in range(valid_lens.size(0)):
            valid_len_value = int(valid_lens[i].item())
            valid_mask[i, valid_len_value:] = 1  # 将前 valid_len 个元素设为 1
        valid_mask = valid_mask.to(tokens.device)
        
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        X = self.blks(X, src_key_padding_mask=valid_mask)
        return X
    

class MaskLM(nn.Module):
    """BERT的掩蔽语⾔模型任务"""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))
        
    def forward(self, X, pred_positions):
        """
        对输入X中的特定位置进行前向传播。

        Args:
            X (torch.Tensor): 输入张量，形状为 (batch_size, seq_length, feature_dim)。
            pred_positions (torch.Tensor): 需要预测的位置张量，形状为 (batch_size, num_pred_positions)。

        Returns:
            torch.Tensor: 预测结果张量，形状为 (batch_size, num_pred_positions, feature_dim)。

        """
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1,1]），将band_idx中的每个数重复num_pred_positions次，
        # [batch_size]->[batch_size*num_pred_positions]
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


class NextSentencePred(nn.Module):
    """BERT的下⼀句预测任务"""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)
        
    def forward(self, X):
        # X的形状：(batchsize,num_hiddens)
        return self.output(X)


class BERTModel(nn.Module):
    """BERT模型"""
    def __init__(self, vocab_size, num_hiddens, ffn_dim,
                num_heads, num_layers, dropout,
                max_len=1000, hid_in_features=768, mlm_in_features=768,
                nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, ffn_dim, num_heads, 
                                   num_layers, dropout, max_len)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)
        
    def forward(self, tokens, segments, valid_lens=None,
                pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # ⽤于下⼀句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat


#-------------------------------------------------#
#   main函数
#-------------------------------------------------#
if __name__ == '__main__':
    vocab_size, num_hiddens, num_heads = 10000, 768, 4
    ffn_dim, num_layers, dropout = 1024, 2, 0.2
    encoder = BERTEncoder(vocab_size, num_hiddens, ffn_dim, num_heads, num_layers, dropout)
    tokens = torch.randint(0, vocab_size, (2, 8))
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
    valid_lens = torch.tensor([6, 6])
    encoded_X = encoder(tokens, segments, valid_lens)
    # print(encoder)
    print(encoded_X.shape)
    
    # mask_lm
    mlm = MaskLM(vocab_size, num_hiddens)
    mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
    mlm_Y_hat = mlm(encoded_X, mlm_positions)
    print(mlm_Y_hat.shape)
    mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
    loss = nn.CrossEntropyLoss(reduction='none')
    mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
    print(mlm_l.shape)

    # next_sentence_pred
    encoded_X = torch.flatten(encoded_X, start_dim=1)
    print(encoded_X.shape)
    # NSP的输⼊形状:(batchsize，num_hiddens)
    nsp = NextSentencePred(encoded_X.shape[-1])
    nsp_Y_hat = nsp(encoded_X)
    print(nsp_Y_hat.shape)
    nsp_y = torch.tensor([0, 1])
    nsp_l = loss(nsp_Y_hat, nsp_y)
    print(nsp_l.shape)
    
        