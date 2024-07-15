# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   dataset.py
# @Time    :   2023/12/11 21:27:57
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   最简单的目标检测DataLoader实现


from typing import Any
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import os
import torchvision
from torch.utils.data import DataLoader
import torch
import json
from pycocotools.coco import COCO
import copy


class CocoDataset(Dataset):
    """
        数据集加载类
    """
    def __init__(self, images_path, label_path, img_size, pipeline=None):
        self.images_path = images_path
        self.label_path = label_path
        self.img_size = img_size
        self.pipeline = pipeline
        # 获取标签
        self.coco = COCO(self.label_path)
        self.img_ids = self.coco.getImgIds()
        # 数据预处理
        self.trans_all = torchvision.transforms.Compose([TFs([640, 640])])
        self.trans_img = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


    def __len__(self):
        return len(self.img_ids)


    def __getitem__(self, idx):
        # 实现获取单个样本的逻辑
        # 选择其中一张图片
        img_id = self.img_ids[idx]
        # 获取该图片的信息，因为只索引了一张图片，所以只有0
        img_info = self.coco.loadImgs(img_id)[0]
        # 获取该图片对应的标注ID
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        # 获取图片文件名
        img_name = img_info['file_name']
        # 获取该图片对应的标注信息
        anns = self.coco.loadAnns(ann_ids)
        # 读取图片
        img_item_path = os.path.join(self.images_path, img_name)
        img = Image.open(img_item_path)
        # 数据预处理
        img, anns = self.trans_all((img, anns))
        img = self.trans_img(img)   # 图片处理要在后面
        return img, anns, img_name


class TFs():
    """
        自定义图片大小并，将标签转化到同一个大小的图片上 transforms
    """
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, data):
        """
            data = (img, anns)
        """
        # 改变图片大小
        new_img = data[0].resize(self.image_size)
        # 改变标签到和图片同样大小的尺寸
        width, height = data[0].size
        t_x = self.image_size[0]/width
        t_y = self.image_size[0]/height
        anns = data[1]
        for ann in anns:
            # coco标签的"bbox": [x,y,width,height],
            # 图片默认宽的方向为x轴，高的方向为y轴。
            ann['new_bbox'] = [0, 0, 0, 0]
            ann['new_bbox'][0] = ann['bbox'][0]*t_x
            ann['new_bbox'][1] = ann['bbox'][1]*t_y
            ann['new_bbox'][2] = ann['bbox'][2]*t_x
            ann['new_bbox'][3] = ann['bbox'][3]*t_y
        return (new_img, anns)


def xywh_xyxy(bbox):
    """
        将左上角和宽高坐标转化为左上角和右下角坐标
    """
    list = copy.deepcopy(bbox)
    list[2] = list[0]+list[2]
    list[3] = list[1]+list[3]
    return list


def img_draw_bbox(data):
    """
        绘制经过数据预处理之后的标签数据
    """
    img, anns = data
    # 创建一个可绘制对象
    draw = ImageDraw.Draw(img)
    # 绘制矩形框
    for ann in anns:
        if 'new_bbox' in ann:
            draw.rectangle(xywh_xyxy(ann['new_bbox']), outline='red', width=2)
        else:
            draw.rectangle(xywh_xyxy(ann['bbox']), outline='red', width=2)
    # 显示图片
    img.show()


def my_collate_fn(batch):
    """
        迭代合并batch自定义处理函数
        img （batch, 3, width,height）
        label （在这个batch中的图片索引序号，类别索引号，x, y, width, height）
    """
    # 拼接图片为一个batch
    img = []
    img_name = []
    # 获取batch中最长数据的长度
    max_length = max(len(data[1]) for data in batch)
    # 创建标签的tensor矩阵
    label = torch.zeros([len(batch), max_length, 6])
    # 将图片数据和标签数据进行拆分，并保存为特定格式
    for i, data in enumerate(batch):
        img.append(data[0])
        img_name.append(data[2])
        for j, ann in enumerate(data[1]):
            y_label = torch.zeros(6)
            y_label[0] = i
            y_label[1] = ann['category_id']
            y_label[2:] = torch.tensor(ann['new_bbox'])
            label[i][j] = y_label
    # 转tensor
    img = torch.stack(img)  # 将tensor列表转化为tensor矩阵
    return img, label, img_name   # 都为tensor矩阵可直接使用


def my_data_load(root_dir, mode='train', img_size=[640, 640], batch_size=1, workers=1, data_enhance=None):
        """
            最终获取数据集迭代对象
        """
        # image_path = root_dir + "\\" + mode + "\\images"
        # label_path = root_dir + "\\" + mode + "\\annotations\\" + mode + ".json"
        image_path = 'H:\\code\\datasets\\coco128\\images\\train2017'
        label_path = 'H:\\code\\datasets\\coco128\\annotations\\train.json'
        dataset = CocoDataset(image_path, label_path, img_size)
        train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=my_collate_fn)
        return train_iter


if __name__ == '__main__':
    # data = CocoDataset("H:\\code\\datasets\\coco128\\images\\train2017", 
    #                    "H:\\code\\datasets\\coco128\\annotations\\train.json", [640, 640])
    # img_draw_bbox(data.__getitem__(1))
    dataloder = my_data_load("H:\\code\\datasets\\TS_dataset4", mode="val")
    
    """
        显示加载的数据
    """
    # for (img, label) in dataloder:   # (img, anns)
    #     # print(img.shape)
    #     # print(label.shape)
    #     for i, image in enumerate(img):
    #         img_pil = torchvision.transforms.ToPILImage()(image)
    #         draw = ImageDraw.Draw(img_pil)
    #         draw.rectangle(xywh_xyxy(label[i][0][2:].tolist()), outline='red', width=2)
    #         img_pil.show()
    #         print()