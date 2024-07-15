import contextlib
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import math
import yaml
import matplotlib as plt
import numpy as np
import thop
import time

from yolov8_model_load.modules import (AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
                                    Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv, DWConvTranspose2d,
                                    Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, Pose, RepC3, RepConv,
                                    RTDETRDecoder, Segment, CNeB, CBAM, SPPCSPC, SPPFCSPC, SwinStage, PatchMerging,
                                    PatchEmbed)

"""
    类
"""

class Model_v8(nn.Module):
    def __init__(self, cfg='yolov8s.yaml', ch=3, nc=None, verbose=True):
        super().__init__()
        self.yaml = load_yaml_v8(cfg)  # cfgy因是完整路径
        self.yaml['ch'] = ch
        self.yaml['nc'] = nc
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        
        # 外部赋值
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}
        self.stride = torch.Tensor([8, 16, 32])
        self.model[-1].stride = self.stride

        # self.inplace是一个布尔值，用于控制是否在原地进行操作。
        # 在深度学习中，原地操作指的是直接修改输入数据，而不是创建新的数据。
        self.inplace = self.yaml.get('inplace', True)
        # 权重初始化
        initialize_weights(self)

        # 打印模型参数信息
        print(f"模型层数: {len(list(self.modules()))}, 模型参数: {sum(x.numel() for x in self.parameters())};")

    def forward(self, x, augment=False, profile=False, visualize=False):
        """
        1. x：这是输入到模型的数据，通常是一个张量（Tensor）。
        2. augment：这是一个布尔值，用于控制是否对输入数据x进行数据增强。
        3. profile：这是一个布尔值，用于控制是否对模型的前向传播过程进行性能分析。
        4. visualize：这是一个布尔值，用于控制是否对模型的前向传播过程进行可视化。
        """
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train
    
    def _forward_augment(self, x):
        """
        函数是用于数据增强的前向传播。
        x 输入图像
        """
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for index, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x
    
    def _profile_one_layer(self, m, x, dt):
        """
        函数的作用是对模型的单个层进行性能分析。
        1. m：这是当前要进行性能分析的模型层。
        2. x：这是输入到模型层m的数据。
        3. dt：这是一个列表，用于保存每一层的运行时间。
        """
        # c = isinstance(m, ACDetect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            print(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        print(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        # if c:
            # print(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _descale_pred(self, p, flips, scale, img_size):
        """
        _descale_pred函数的作用是对经过数据增强（如缩放和翻转）后的预测结果进行反向操作
        1. p：这是模型的预测结果，通常是一个张量（Tensor），包含了边界框的坐标和大小等信息。
        2. flips：这是一个整数，表示在数据增强时进行的翻转操作。如果flips=2，表示进行了上下翻转；
        3. scale：这是一个浮点数，表示在数据增强时进行的缩放操作。
        4. img_size：这是一个元组，表示原始图像的尺寸（高度和宽度）。
        """
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p
    
    def _clip_augmented(self, y):
        """
        函数的作用是对经过数据增强后的预测结果进行裁剪。在数据增强的过程中，
        可能会生成一些超出原始图像范围的预测结果，这些结果通常是无效的，需要被裁剪掉。
        """
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y
    

"""
    函数
"""

def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def feature_visualization(x, module_type, stage, n=32, save_dir=Path('runs/detect/exp')):
    """
    x:              Features to be visualized
    module_type:    Module type
    stage:          Module stage within model
    n:              Maximum number of feature maps to plot
    save_dir:       Directory to save results
    """
    if 'Detect' not in module_type:
        batch, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = save_dir / f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # filename

            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
            n = min(n, channels)  # number of plots
            fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                ax[i].axis('off')

            print(f'Saving {f}... ({n}/{channels})')
            plt.savefig(f, dpi=300, bbox_inches='tight')
            plt.close()
            np.save(str(f.with_suffix('.npy')), x[0].cpu().numpy())  # npy save


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean
    

def initialize_weights(model):
    """
        模型权重初始化
    """
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def load_yaml_v8(path):
    """
        加载yaml文件
    """
    path_ture = path[:-6]+".yaml"   # 需要去掉模型规格那个字母才能找到文件
    if isinstance(path_ture, str):    
        with open(path_ture, 'r', encoding='utf-8') as f:
            data = yaml.load(f.read(), Loader=yaml.FullLoader)
    data['scale'] = path[-6]
    data['yaml_file'] = path
    return data


def make_divisible(x, divisor):
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """
    函数功能：解析模型
    :param d:模型配置字典
    :param ch:模型输入通道数
    :param verbose:是否在运行时输出详细的模型结构
    :return:
    """
    # Parse a YOLO model.yaml dictionary into a PyTorch model
    import ast      # 检查、修改和生成Python代码的AST

    # Args
    max_channels = float('inf')     # 定义最大通道数
    nc, act, scales = (d.get(x) for x in ('nc', 'activation', 'scales'))    # 读取配置文件中的这三个参数，activation代表激活函数，在yolov6中使用
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))    # 获取参数，kpt_shape在姿态估计中才使用
    # 读取模型缩放参数
    if scales:
        scale = d.get('scale')
        """
        如果未指定模型scale, 默认为 n，
        scales.keys(['n','s','m','l','x'])
        """
        if not scale:
            scale = tuple(scales.keys())[0]
            print(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]
    # 修改激活函数
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            print(f"{colorstr('activation:')} {act}")  # print

    if verbose: # 输出一些日志
        print(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]   # 将输入通道转化为列表
    """
    layers 是将要添加到模型的层
    save 中间层的索引列表
    c2 输出通道数
    """
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    """
        循环遍历所有层，生成网络
    """
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        """
        检查字符串变量中是否包含 nn
        如果有，那么从torch.nn中获取与该字符串对应的属性或类
        如果没有则从当前的全局命名空间当中查找与m对应的值 globals()[m]，就是在ultralytics/nn/modules下寻找
        """
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
        # 解析 args 参数，就是每个模块需要传入的参数，将参数由字符串转化为实际的可用参数
        for j, a in enumerate(args):
            if isinstance(a, str):  # 检查是否是字符串类型
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        # 模型深度调整，这里是调整每个块的数量
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain

        """
        针对不同的模块处理成不同的参数输入，输入形式相同的可以归为一类
        自己设计模块之后只需要建立对应的 elif 分支即可，如果输入参数顺序可以和其他模块共用，也可以直接添加在其他分支
        c1 : 输入通道数
        c2 : 输出通道数
        args : 配置文件传入的参数
        args = [c1, c2, *args[1:]]  处理好参数之后，重新将参数放到 args 中即可
        args.insert(2, n)  也可以通过插入语句在列表中插入元素
        """
        if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                 BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3,
                 SPPCSPC, SPPFCSPC):
            c1, c2 = ch[f], args[0]     # 输入通道数，输出通道数
            # 如果不是模型输入通道，要与模型宽度缩放系数相乘，对模型进行缩放，并且缩放之后的输入通道数要必须是8的倍数，不是则转化为8的倍数
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]      # 最终传递给该模块的参数
            # 如果是以下模块就在args列表的2位置插入模块个数这个参数
            if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3):
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in (HGStem, HGBlock):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in (Detect, Segment, Pose, RTDETRDecoder):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)     # 添加模块，再次后面添加一个 elif 分支即可
        elif m is CNeB:     # 自己添加的
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels)*width, 8)

            args = [c1, c2, *args[1:]]
            if m is CNeB:
                args.insert(2, n)
                n = 1
        elif m is CBAM:     # 自己添加的
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, *args[1:]]
        elif m in [SwinStage, PatchMerging, PatchEmbed]:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
        else:
            c2 = ch[f]

        # 根据模块重复的次数添加模块
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module

        # 取出模块的名称
        t = str(m)[8:-2].replace('__main__.', '')  # module type

        # 计算模块的总参数量
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        # 获取一些额外信息
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        # 打印输出模型信息
        if verbose:
            print(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')  # print
        # 保存模型信息和模型
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # 存储需要保存输出的层号
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)   # 存储模型输出通道数
    return nn.Sequential(*layers), sorted(save)

