# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   utils.py
# @Time    :   2024/11/05 11:45:32
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   新创建的python脚本

import time
import numpy as np
from IPython import display
from matplotlib_inline import backend_inline
from matplotlib import pyplot as plt
import random

import torch


class Timer:
    """记录多次运⾏时间"""
    def __init__(self):
        self.times = []
        self.start()
        
    def start(self):
        """启动计时器"""
        self.tik = time.time()
        
    def stop(self):
        """停⽌计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    
    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)
    
    def sum(self):
        """返回时间总和"""
        return sum(self.times)
    
    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()
    
    
def use_svg_display():
    """使⽤svg格式在Jupyter中显⽰绘图"""
    backend_inline.set_matplotlib_formats('svg')


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


class Animator: #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使⽤lambda函数捕获参数
        self.config_axes = lambda: set_axes(
        self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        
    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        # 保存图形为图片文件
        self.fig.savefig(f'loss.png')  # 保存为不同的文件名
        display.clear_output(wait=True)  # 清除输出
 

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n
        
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
        
    def reset(self):
        self.data = [0.0] * len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

   
def load_safetensors_weights(path, device="cuda"):
    from safetensors.torch import load_file
    return load_file(path, device=device)


def save_safetensors_weights(path, weights):
    from safetensors.torch import save_file
    save_file(weights, path)
        

def load_bert_base_weights(weight_path):
    """
    加载BERT基础权重,并去除"bert."前缀，基于bert微调时使用。

    Args:
        weight_path (str): BERT基础权重的文件路径

    Returns:
        dict: 去除"bert."前缀的权重字典

    """
    weights = torch.load(weight_path)
    new_weights = {}
    for key in weights.keys():
        if "cls" not in key and "pooler" not in key:
            newkey = key.replace("bert.", "")
            new_weights[newkey] = weights[key]
    return new_weights


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
    
#-------------------------------------------------#
#   main函数
#-------------------------------------------------#
if __name__ == '__main__':
    import torch
    weights = load_safetensors_weights('/media/lht/D_Project/models/bert-large-English/model.safetensors')
    weights = {k: v.to("cpu") for k, v in weights.items()}
    weights2 = torch.load('/media/lht/D_Project/models/bert-large-English/pytorch_model.bin')
    print(weights.keys())
    print("-------------------------------")
    print(weights2.keys())
    