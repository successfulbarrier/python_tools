# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   model.py
# @Time    :   2023/12/12 16:59:20
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   网络模型实现

from torch import nn
from torch.nn import functional as F
import torch
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import math
from loss import LOSS, GenTargets
from inference import DetectHead, ClipBoxes
from cfg import DefaultConfig


"""
    主干网络，resnet
"""
class ResNet(nn.Module):
    def __init__(self, size="resnet50", pretrained=True) -> None:
        super(ResNet, self).__init__()
        # 加载预训练的 resnet50 模型
        if size == "resnet18":
            # 网上下载权重 weights=models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=None)
            if pretrained:
                self.backbone.load_state_dict(torch.load('./checkpoint/resnet18.pth'),strict=True)
        elif size == "resnet34":
            self.backbone = models.resnet34(weights=None)
            if pretrained:
                self.backbone.load_state_dict(torch.load('./checkpoint/resnet34.pth'),strict=True)
        elif size == "resnet50":
            self.backbone = models.resnet50(weights=None)
            if pretrained:
                self.backbone.load_state_dict(torch.load('./checkpoint/resnet50.pth'),strict=True)
        elif size == "resnet101":
            self.backbone = models.resnet101(weights=None)
            if pretrained:
                self.backbone.load_state_dict(torch.load('./checkpoint/resnet101.pth'),strict=True)
        elif size == "resnet152":
            self.backbone = models.resnet152(weights=None)
            if pretrained:
                self.backbone.load_state_dict(torch.load('./checkpoint/resnet152.pth'),strict=True)
        else:
            raise Exception("resnet网络size设置错误！！！！")   # 抛出异常
            
        # 移除最后一层全连接层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])   # 具体如何裁剪要看具体模型和自己需求

        # 计算输出层的通道数
        self.out_channel = [0, 0, 0]
        self.out_channel[0], self.out_channel[1], self.out_channel[2] = self.get_out_channel()

    def forward(self, x):
        # 获取网络的多层输出
        out1 = self.backbone[:-2](x)
        out2 = self.backbone[-2](out1)
        out3 = self.backbone[-1](out2)
        return (out1, out2, out3)
    
    def freeze_bn(self):    # 冻结BN层
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                # print(layer)
                layer.eval()
    
    def get_out_channel(self):
        input_data = torch.randn(1, 3, 640, 640, dtype=torch.float32)
        (out1, out2, out3) = self.forward(input_data)
        return out1.shape[1], out2.shape[1], out3.shape[1]


"""
    颈部网络
"""
class FPN(nn.Module):
    '''only for resnet50,101,152'''
    def __init__(self,features=256,use_p5=True,BO_channel=[512, 1024, 2048]):
        super(FPN,self).__init__()
        self.prj_5 = nn.Conv2d(BO_channel[2], features, kernel_size=1)
        self.prj_4 = nn.Conv2d(BO_channel[1], features, kernel_size=1)
        self.prj_3 = nn.Conv2d(BO_channel[0], features, kernel_size=1)
        self.conv_5 =nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_4 =nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_3 =nn.Conv2d(features, features, kernel_size=3, padding=1)
        
        # FCOS是用的P5产生P6的
        if use_p5:
            self.conv_out6 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)  # 使用P5的输出
        else:
            self.conv_out6 = nn.Conv2d(BO_channel[2], features, kernel_size=3, padding=1, stride=2) # 使用主干网络的最后一层的输出
        self.conv_out7 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        self.use_p5=use_p5
        self.apply(self.init_conv_kaiming)  # 初始化参数

    def upsamplelike(self,inputs):
        src,target=inputs
        # 最近邻上采样，src是要上采样的数据，target是要上采样到的大小
        # interpolate()函数可选的上采样方式包括：最近邻插值（'nearest'）、双线性插值（'bilinear'）、
        # 双三次插值（'bicubic'）、面积插值（'area'）、像素随机值插值（'pixelshuffle'）等。
        return F.interpolate(src, size=(target.shape[2], target.shape[3]),
                    mode='nearest')
    
    def init_conv_kaiming(self,module):
        # 初始化卷积核的参数
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self,x):
        C3,C4,C5=x
        P5 = self.prj_5(C5)
        P4 = self.prj_4(C4)
        P3 = self.prj_3(C3)
        
        P4 = P4 + self.upsamplelike([P5,C4])
        P3 = P3 + self.upsamplelike([P4,C3])

        P3 = self.conv_3(P3)
        P4 = self.conv_4(P4)
        P5 = self.conv_5(P5)
        
        P5 = P5 if self.use_p5 else C5
        P6 = self.conv_out6(P5)
        P7 = self.conv_out7(F.relu(P6))
        return [P3,P4,P5,P6,P7]


"""
    头部网络
"""
# 因为regress的ltrb总是为正值，所以要做一个exp保证其为正值
class ScaleExp(nn.Module):
    def __init__(self,init_value=1.0):
        super(ScaleExp,self).__init__()
        self.scale=nn.Parameter(torch.tensor([init_value],dtype=torch.float32))
    def forward(self,x):
        return torch.exp(x*self.scale)

class ClsCntRegHead(nn.Module):
    # GN 就是groupnorm，原文中有说用这个会有效果，centerness分支加到reg也是有更好的效果的
    def __init__(self,in_channel,class_num,GN=True,cnt_on_reg=True,prior=0.01):
        '''
        Args  
        in_channel  
        class_num  
        GN  
        prior  
        '''
        super(ClsCntRegHead,self).__init__()
        self.prior=prior
        self.class_num=class_num
        self.cnt_on_reg=cnt_on_reg

        # 4次不改变tensor的卷积,所有特征层共用这四层卷积
        # 共用公共卷积可以减少模型参数量，提高模型的训练速度和泛化能力，但可能会限制模型对不同特征层的适应性。
        # 不共用公共卷积可以增加模型的灵活性，使得模型能够更好地适应不同特征层的特征表达，但也会增加模型的参数量和训练时间。
        cls_branch=[]
        reg_branch=[]

        for i in range(4):
            cls_branch.append(nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1,bias=True))
            # 是否用GroupNorm.在head部分用GN有效果
            if GN:
                cls_branch.append(nn.GroupNorm(32,in_channel))      # 组归一化有助于加速训练并提高模型的泛化能力
            cls_branch.append(nn.ReLU(True))

            reg_branch.append(nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1,bias=True))
            if GN:
                reg_branch.append(nn.GroupNorm(32,in_channel))
            reg_branch.append(nn.ReLU(True))

        self.cls_conv=nn.Sequential(*cls_branch)
        self.reg_conv=nn.Sequential(*reg_branch)
        # 最后输出的tensor
        self.cls_logits=nn.Conv2d(in_channel,class_num,kernel_size=3,padding=1)
        self.cnt_logits=nn.Conv2d(in_channel,1,kernel_size=3,padding=1)
        self.reg_pred=nn.Conv2d(in_channel,4,kernel_size=3,padding=1)

        # 以下都是初始化参数
        self.apply(self.init_conv_RandomNormal)
        # prior的使用
        # nn.init.constant_() 函数的作用是将输入的张量参数初始化为指定的常数值。
        nn.init.constant_(self.cls_logits.bias,-math.log((1 - prior) / prior))
        self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(5)])
    
    def init_conv_RandomNormal(self,module,std=0.01):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=std)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self,inputs):
        '''inputs:[P3~P7]'''
        cls_logits=[]
        cnt_logits=[]
        reg_preds=[]
        for index,P in enumerate(inputs):
            cls_conv_out=self.cls_conv(P)
            reg_conv_out=self.reg_conv(P)

            cls_logits.append(self.cls_logits(cls_conv_out))
            # cnt在reg效果更好
            if not self.cnt_on_reg:
                cnt_logits.append(self.cnt_logits(cls_conv_out))
            else:
                cnt_logits.append(self.cnt_logits(reg_conv_out))
            reg_preds.append(self.scale_exp[index](self.reg_pred(reg_conv_out)))
        return cls_logits,cnt_logits,reg_preds
    

"""
    整个网络实现
"""
class FCOS(nn.Module):
    
    def __init__(self,config=None):
        super().__init__()
        if config is None:  
            config=DefaultConfig
        self.backbone=ResNet(size=config.backbone_name)
        self.fpn=FPN(config.fpn_out_channels,use_p5=config.use_p5,BO_channel=self.backbone.out_channel)
        self.head=ClsCntRegHead(config.fpn_out_channels,config.class_num,
                                config.use_GN_head,config.cnt_on_reg,config.prior)
        self.config=config

    def train(self,mode=True):
        '''
        set module training mode, and frozen bn
        '''
        super().train(mode=True)
        # 训练的时候我们一般把第一层给Freeze。
        def freeze_bn(module):
            if isinstance(module,nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters(): p.requires_grad=False
        if self.config.freeze_bn:
            self.apply(freeze_bn)
            print("INFO===>success frozen BN")
        if self.config.freeze_stage_1:
            self.backbone.freeze_stages(1)
            print("INFO===>success frozen backbone stage1")

    def forward(self,x):
        '''
        Returns
        list [cls_logits,cnt_logits,reg_preds]  
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        '''
        C3,C4,C5=self.backbone(x)
        all_P=self.fpn([C3,C4,C5])
        cls_logits,cnt_logits,reg_preds=self.head(all_P)
        return [cls_logits,cnt_logits,reg_preds]


"""
    整合loss之后的模型
"""
class FCOSDetector(nn.Module):
    def __init__(self,mode="training",config=None):
        super().__init__()
        if config is None:
            config=DefaultConfig
        self.mode=mode
        self.fcos_body=FCOS(config=config)
        if mode=="training":
            self.target_layer=GenTargets(strides=config.strides,limit_range=config.limit_range)
            self.loss_layer=LOSS()
        elif mode=="inference":
            self.detection_head=DetectHead(config.score_threshold,config.nms_iou_threshold,
                                            config.max_detection_boxes_num,config.strides,config)
            self.clip_boxes=ClipBoxes()
        
    def forward(self,inputs):
        '''
        inputs 
        [training] list  batch_imgs,batch_boxes,batch_classes
        [inference] img
        '''

        if self.mode=="training":
            batch_imgs,batch_boxes,batch_classes=inputs
            out=self.fcos_body(batch_imgs)
            targets=self.target_layer([out,batch_boxes,batch_classes])
            losses=self.loss_layer([out,targets])
            return losses
        elif self.mode=="inference":
            # raise NotImplementedError("no implement inference model")
            '''
            for inference mode, img should preprocessed before feeding in net 
            '''
            batch_imgs=inputs
            out=self.fcos_body(batch_imgs)
            scores,classes,boxes=self.detection_head(out)
            boxes=self.clip_boxes(batch_imgs,boxes)
            return scores,classes,boxes


if __name__ == '__main__':
    model_list = dir(models.detection)  # 获取pytorch提供的检测模型列表
    print(model_list)
    fcos = models.detection.fcos.fcos_resnet50_fpn(pretrained=True)

