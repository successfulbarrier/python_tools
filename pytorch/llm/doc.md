# 大语言模型基础

## 1.TransformerEncoder
使用pytorch.nn模块中的TransformerEncoder和TransformerEncoderLayer构建一个简单的Transformer编码器。

官方文档：https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html#torch.nn.TransformerEncoder ，https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html#torch.nn.TransformerEncoderLayer

示例代码：只传入了必要的参数，更多参数请参考官方文档
```python
import torch
from torch import nn
tokens = torch.rand(4, 32, 512)
valid_lens = torch.tensor([31, 24, 16, 8])  # 输入序列的有效长度
# valid_lens转化填充位置为掩码
valid_mask = torch.zeros((valid_lens.size(0), tokens.shape[1]), dtype=torch.bool)  # 初始化全为0的二维 Tensor
for i in range(valid_lens.size(0)):
    valid_len_value = int(valid_lens[i].item())
    valid_mask[i, valid_len_value:] = 1  # 将前 valid_len 个元素设为 1
valid_mask = valid_mask.to(tokens.device)
print(valid_mask.shape)
transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(512, 8, batch_first=True), num_layers=6)
memory = transformer_encoder(tokens, src_key_padding_mask=valid_mask)
print(memory.shape)
```
说明：
memory.shape 输出为 torch.Size([4, 32, 512])
valid_mask.shape 输出为 torch.Size([4, 32])

## 2.文本编码 Embedding
使用pytorch.nn模块中的Embedding构建一个简单的文本Embedding。

官方文档：https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding

示例代码：只传入了必要的参数，更多参数请参考官方文档
```python
import torch
from torch import nn
vocab_size = 10000
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=768)
tokens = torch.randint(0, vocab_size, (2, 8))
print(embedding(tokens))
```
说明：nn.Embedding的num_embeddings规定了传入tokens的词表大小，embedding_dim规定了Embedding的维度。
这里我们的词表大小为10000，因此tokens中的数值范围为0-9999，经过Embedding后，输出为torch.Size([2, 8, 768]),将每个词由一个数字转化为了768维的向量，更有利于训练。


## 3.位置编码 Positional Encoding
自定义一个矩阵实现位置编码。

示例代码：
```python
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    '''
    位置编码

    Args:
        max_len: 位置编码矩阵的最大长度,代表字符串的最大长度
        num_hiddens: 隐藏层的维度和transformer编码器的维度相同,一般设置为768

    '''
    def __init__(self, max_len=1000, num_hiddens=768):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))  # 位置编码矩阵[1, 1000, 768]

    def forward(self, x):
        x = x + self.pos_embedding.data[:, :x.shape[1], :]
        return x
```
说明：该位置编码的用法，为在前向传播中，将位置编码矩阵和经过Embedding后的tokens相加，实现位置编码。然后将相加后的tokens输入Transformer编码器中。
例如：
```python
X = self.token_embedding(tokens) + self.segment_embedding(segments)
X = self.pos_embedding(X)
X = self.encoder(X, valid_lens)
```

## 4.取出tensor中特定位置的元素
示例代码：
```python
import torch
X = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
select_X = torch.tensor([[0, 1], [1, 2], [0, 2]])
batch_idx = torch.arange(0, X.shape[0])
batch_idx = torch.repeat_interleave(batch_idx, select_X.shape[1])
select_X = X[batch_idx, select_X.reshape(-1)]
select_X = select_X.reshape((X.shape[0], 2, -1))
print(select_X)
```
说明：torch.repeat_interleave() 是 PyTorch 中的一个函数，用于在指定维度上重复张量的元素。该函数将张量中的每个元素重复指定的次数，并将结果拼接在一起。


## 5.使用tramformers库预训练bert模型
20行代码预训练bert模型

示例代码：
```python
import os
import sys
sys.path.append("./")
import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertForPreTraining
from transformers.models.bert.configuration_bert import BertConfig
from transformers import BertTokenizer

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
    loss = bert_model(input_ids=text_tesnsor["input_ids"].to(devices),                  
                    attention_mask=text_tesnsor["attention_mask"].to(devices),
                    labels=torch.tensor([[-100, -100, -100, 1037, -100, -100],
                                         [-100, -100, -100, 1037, -100, -100],
                                         [-100, -100, -100, 1037, -100, -100]]).to(devices), 
                    next_sentence_label=torch.tensor([0, 0, 0]).to(devices), return_dict=False)
    print(loss[0])
    loss[0].backward()
    trainer.step()
```
说明：使用一句话作为数据示例，训练10次，打印loss值。
