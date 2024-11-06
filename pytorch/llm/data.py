# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   data.py
# @Time    :   2024/11/04 19:58:08
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   大语言模型数据集处理

import os
import random
import torch
import collections


#-------------------------------------------------#
#   在WikiText-2数据集中，每⾏代表⼀个段落，其中在任意标点符号及其前⾯的词元之间插⼊空格。保留⾄少
#   有两句话的段落。为了简单起⻅，我们仅使⽤句号作为分隔符来拆分句⼦。
#-------------------------------------------------#
def _read_wiki(data_dir):
    """
    从指定目录中读取wiki训练数据，并返回处理后的段落列表。

    Args:
        data_dir (str): 存储wiki训练数据的目录路径。

    Returns:
        list of list of str: 处理后的段落列表，每个段落为一个包含多个句子的列表。

    """
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # ⼤写字⺟转换为⼩写字⺟
    paragraphs = [line.strip().lower().split(' . ')
    for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs


def _get_next_sentence(sentence, next_sentence, paragraphs):
    """
    从给定的段落中随机选择一个句子作为下一个句子。

    Args:
        sentence (str): 当前句子。
        next_sentence (str): 当前句子的下一个句子。
        paragraphs (list of list of list of str): 包含多个段落的列表，每个段落是句子的列表。

    Returns:
        tuple: 包含当前句子、随机选择的下一个句子以及表示是否是下一个句子的布尔值。
    """
    if random.random() < 0.5:
        is_next = True
    else:
        # paragraphs是三重列表的嵌套
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """
    获取输入序列的词元及其片段索引。

    Args:
        tokens_a (List[str]): 第一个输入序列的词元列表。
        tokens_b (List[str], optional): 第二个输入序列的词元列表，默认为None。

    Returns:
        Tuple[List[str], List[int]]: 包含词元和片段索引的元组。
            - tokens (List[str]): 输入序列的词元列表，包括 '<cls>' 和 '<sep>' 标记。
            - segments (List[int]): 每个词元对应的片段索引列表，0 表示第一个片段，1 表示第二个片段。

    """
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分别标记⽚段A和B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
    segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    """
    从段落中提取NSP（Next Sentence Prediction）数据。

    Args:
        paragraph (list of str): 一个字符串列表，表示单个段落。
        paragraphs (list of list of str): 一个字符串列表的列表，表示多个段落。
        vocab (dict): 词汇表，用于将字符串转换为索引。
        max_len (int): 序列的最大长度。

    Returns:
        list of tuple: 包含NSP数据的列表，每个元素都是一个包含tokens、segments和is_next的元组。

    """
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # 考虑1个'<cls>'词元和2个'<sep>'词元
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph


def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    """
    为遮蔽语⾔模型的输⼊创建新的词元副本，其中输⼊可能包含替换的“<mask>”或随机词元。

    Args:
        tokens (list): 原始词元列表。
        candidate_pred_positions (list): 候选预测位置列表。
        num_mlm_preds (int): 预测的遮蔽语⾔模型数量。
        vocab (Vocab): 词汇表对象。

    Returns:
        tuple: 包含两个元素的元组，
            - mlm_input_tokens (list): 处理后的词元列表。
            - pred_positions_and_labels (list): 包含预测位置和标签的列表。

    """
    # 为遮蔽语⾔模型的输⼊创建新的词元副本，其中输⼊可能包含替换的“<mask>”或随机词元
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # 打乱后⽤于在遮蔽语⾔模型任务中获取15%的随机词元进⾏预测
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80%的时间：将词替换为“<mask>”词元
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10%的时间：保持词不变
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10%的时间：⽤随机词替换该词
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
        (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def _get_mlm_data_from_tokens(tokens, vocab):
    """
    从给定的token列表中获取用于遮蔽语言模型（Masked Language Model，MLM）任务的输入数据和预测位置。

    Args:
        tokens (List[str]): 一个字符串列表，代表文本中的token。
        vocab (Dict[str, int]): 词汇表，用于将token映射到对应的整数索引。

    Returns:
        Tuple[List[int], List[int], List[int]]:
            - vocab[mlm_input_tokens] (List[int]): 经过遮蔽处理后的token列表对应的整数索引列表。
            - pred_positions (List[int]): 需要预测的token在原始token列表中的位置索引列表。
            - vocab[mlm_pred_labels] (List[int]): 需要预测的token对应的真实标签（整数索引）列表。

    """
    candidate_pred_positions = []
    # tokens是⼀个字符串列表
    for i, token in enumerate(tokens):
        # 在遮蔽语⾔模型任务中不会预测特殊词元
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 遮蔽语⾔模型任务中预测15%的随机词元
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def _pad_bert_inputs(examples, max_len, vocab):
    """
    对BERT输入进行填充处理。

    Args:
        examples (tuple): 包含token_ids, pred_positions, mlm_pred_label_ids, segments, is_next的元组列表。
            - token_ids (list of int): 文本经过分词后的token ID列表。
            - pred_positions (list of int): 需要预测的token位置的索引列表。
            - mlm_pred_label_ids (list of int): 需要预测的token的真实label ID列表。
            - segments (list of int): 句子段标识列表，用于区分不同的句子。
            - is_next (bool): 指示下一个句子是否是文本中紧随其后的句子。
        max_len (int): 输入的最大长度。
        vocab (dict): 分词器的词汇表。

    Returns:
        tuple: 包含填充后的token ID列表、句子段标识列表、有效长度列表、预测位置列表、MLM权重列表、MLM标签列表和NSP标签列表的元组。
            - all_token_ids (list of torch.Tensor): 填充后的token ID列表。
            - all_segments (list of torch.Tensor): 填充后的句子段标识列表。
            - valid_lens (list of torch.Tensor): 有效长度列表，不包含填充符号'<pad>'。
            - all_pred_positions (list of torch.Tensor): 填充后的预测位置列表。
            - all_mlm_weights (list of torch.Tensor): 填充后的MLM权重列表，填充部分的权重为0。
            - all_mlm_labels (list of torch.Tensor): 填充后的MLM标签列表。
            - nsp_labels (list of torch.Tensor): NSP标签列表。
    """
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens, = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
        is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        # valid_lens不包括'<pad>'的计数
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # 填充词元的预测将通过乘以0权重在损失中过滤掉
        all_mlm_weights.append(torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
            max_num_mlm_preds - len(pred_positions)),dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
        all_mlm_weights, all_mlm_labels, nsp_labels)


def tokenize(lines, token='word'):
    """
    将文本行拆分为单词或字符词元。

    Args:
        lines (list of str): 要拆分的文本行列表。
        token (str): 指定词元类型，'word' 表示按单词拆分，'char' 表示按字符拆分。

    Returns:
        list of list of str: 拆分后的词元列表。

    Raises:
        ValueError: 如果 token 参数的值不是 'word' 或 'char'，则引发异常。
    """
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)
        

def count_corpus(tokens): 
    """统计词元的频率"""
    # 这⾥的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成⼀个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    """⽂本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
        reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                            for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    @property
    def unk(self):
        # 未知词元的索引为0
        return 0
    
    @property
    def token_freqs(self):
        return self._token_freqs


class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # 输⼊paragraphs[i]是代表段落的句⼦字符串列表；
        # ⽽输出paragraphs[i]是代表段落的句⼦列表，其中每个句⼦都是词元列表
        paragraphs = [tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                    for sentence in paragraph]
        self.vocab = Vocab(sentences, min_freq=5, reserved_tokens=[
        '<pad>', '<mask>', '<cls>', '<sep>'])
        # 获取下⼀句⼦预测任务的数据
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                            paragraph, paragraphs, self.vocab, max_len))
        # 获取遮蔽语⾔模型任务的数据
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                    + (segments, is_next))
                    for tokens, segments, is_next in examples]
        # 填充输⼊
        (self.all_token_ids, self.all_segments, self.valid_lens,
        self.all_pred_positions, self.all_mlm_weights,
        self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
        examples, max_len, self.vocab)    
        
    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
        self.valid_lens[idx], self.all_pred_positions[idx],
        self.all_mlm_weights[idx], self.all_mlm_labels[idx],
        self.nsp_labels[idx])
        
    def __len__(self):
        return len(self.all_token_ids)     
    
    
def load_data_wiki(batch_size, max_len):
    """加载WikiText-2数据集"""
    num_workers = 8
    data_dir = "/media/lht/D_Project/datasets/wikitext-2"
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
    shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab     


#-------------------------------------------------#
#   main函数
#-------------------------------------------------#
if __name__ == '__main__':
    batch_size, max_len = 512, 64
    train_iter, vocab = load_data_wiki(batch_size, max_len)
    for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
        mlm_Y, nsp_y) in train_iter:
        print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
            pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
            nsp_y.shape)
        break
    
          