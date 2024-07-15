#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:python_tools
# author:机灵巢穴_WitNest
# datetime:2023/10/30 9:16
# software: PyCharm
"""
    定义了一些基础类，可以直接继承使用，会非常的方便
"""

import yaml
import json
import shutil
import random
import os
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


"""
    基础类1：关于处理数据集的一些常用方法
"""


class dataset_base(object):
    def __init__(self, yaml_path):
        # 加载配置文件数据
        with open(yaml_path, 'r', encoding='utf-8') as f:
            result = yaml.load(f.read(), Loader=yaml.FullLoader)

        # 设定基础参数
        self.dataset_root = result["path"]
        self.images_path = os.path.join(self.dataset_root, result["images"])     # 图片路径
        self.labels_path = os.path.join(self.dataset_root, result["labels"])     # 标注文件路径
        self.jsons_path = os.path.join(self.dataset_root, result["jsons"])       # json标签文件路径
        self.train_path = os.path.join(self.dataset_root, result["train"])       # 训练集的路径
        self.val_path = os.path.join(self.dataset_root, result["val"])           # 验证集的路径
        self.classes = os.path.join(self.dataset_root, result["classes"])
        self.img_tail = [".jpg", ".jpeg"]   # 图片后缀
        self.label_tail = [".txt", ".json"]     # 标签文件后缀
        self.dataset_save_path = self.dataset_root  # 分好的数据集保存位置
        self.names = result["names"]
        self.class_list = {}
        for k, v in self.names.items():
            self.class_list[v] = k

        # 检查路径
        assert os.path.exists(self.images_path) and os.path.exists(self.jsons_path) and \
               os.path.exists(self.classes), "images, jsons classes 路径有误！"

        if not os.path.exists(self.labels_path):
            os.makedirs(self.labels_path)
        if not os.path.exists(self.train_path):
            os.makedirs(self.train_path)
        if not os.path.exists(self.val_path):
            os.makedirs(self.val_path)

    def json_txt(self):     # json转txt
        # 坐标转化
        def convert(img_size, box):
            dw = 1. / (img_size[0])
            dh = 1. / (img_size[1])
            x = (box[0] + box[2]) / 2.0 - 1
            y = (box[1] + box[3]) / 2.0 - 1
            w = box[2] - box[0]
            h = box[3] - box[1]
            x = x * dw
            w = w * dw
            y = y * dh
            h = h * dh
            return (x, y, w, h)

        # 解码json文件
        def decode_json(json_floder_path, json_name):
            # 获取文件路径
            txt_name = os.path.join(self.labels_path, json_name[0:-5] + '.txt')
            # 存放txt的绝对路径
            txt_file = open(txt_name, 'w')

            json_path = os.path.join(json_floder_path, json_name)
            data = json.load(open(json_path, 'r', encoding='gb2312', errors='ignore'))

            img_w = data['imageWidth']
            img_h = data['imageHeight']

            for i in data['shapes']:
                label_name = i['label']
                if i['shape_type'] == 'rectangle':
                    x1 = int(i['points'][0][0])
                    y1 = int(i['points'][0][1])
                    x2 = int(i['points'][1][0])
                    y2 = int(i['points'][1][1])
                    bb = (x1, y1, x2, y2)
                    bbox = convert((img_w, img_h), bb)
                    txt_file.write(str(self.class_list[label_name]) + " " + " ".join([str(a) for a in bbox]) + '\n')
        json_names = os.listdir(self.jsons_path)
        for json_name in tqdm(json_names, desc="json转txt"):
            decode_json(self.jsons_path, json_name)

    def seek_txt(self, file_list):
        """
        函数功能：寻找列表中的txt文件
        :param file_list: 文件列表
        :return:
        """
        txt_list = []
        for file in file_list:
            if file[-4:] == ".txt":
                txt_list.append(file)
        return txt_list

    def object_num_once(self, file_path, class_list, class_num):
        """
        函数功能：计算单个标签文件中的目标数量
        :param file_path: 文件路径
        :param class_list: 类别列表
        :return:统计好的类别列表
        """
        with open(file_path, 'r') as f:
            while True:
                num = 0
                line = f.readline()
                if line == '':  # 如果文件结束，结束循环
                    break
                for s in line:  # 寻找第一个数的真实长度
                    if s == ' ':
                        break
                    num += 1
                object_class = int(line[0:num])
                for k, v in class_list.items():
                    if class_list[k] == object_class:
                        class_num[k] += 1

    def object_num(self, file_list, root_path, class_list):
        """
        函数功能：计算目标个数
        :param file_list: 文件列表
        :param root_path: 文件路径
        :param class_list: 类别字典
        :return:数量和比例
        """
        # 拷贝类别字典，用于记录目标个数
        class_num = copy.deepcopy(class_list)
        class_ratio = copy.deepcopy(class_list)
        # 初始化目标个数记录字典
        for k, v in class_num.items():
            class_num[k] = 0
        for k, v in class_ratio.items():
            class_ratio[k] = 0

        # 遍历计算目标个数 tqdm(file_list, desc="计算目标数量")
        for file in file_list:
            file_path = os.path.join(root_path, file)
            self.object_num_once(file_path, class_list, class_num)

        # 统计数据
        all_num = 0
        for k, v in class_num.items():
            all_num += class_num[k]
        for k, v in class_num.items():
            class_ratio[k] = round(class_num[k] / all_num, 3)  # 保留三位小数

        return class_num, class_ratio  # 数量，占比

    def plt_table(self, col, row, vals):
        """
        函数功能：使用 matplotlib 绘制表格
        :param col: 列名
        :param row: 行名
        :param vals: numpy列表值
        :return:
        """
        # 支持中文
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.figure(figsize=(30, 60))  # 设置整幅图像大小
        tab = plt.table(cellText=vals,
                        colLabels=col,
                        rowLabels=row,
                        loc='center',
                        cellLoc='center',
                        rowLoc='center')
        tab.scale(0.8, 10)  # 设置格子比例
        tab.set_fontsize(60)  # 设置字体大小
        plt.axis('off')
        plt.gcf().subplots_adjust(left=0.4, top=0.6, bottom=0.4, right=0.6)
        plt.show()

    def save_result(self, col, row, vals, save_format="csv"):
        """
        # 函数功能保存统计的结果
        :param save_format: 保存的格式
        :param col: 列名
        :param row: 行名
        :param vals: numpy列表值
        :return:
        """
        if save_format == "plt_table":
            self.plt_table(col, row, vals)
        else:
            df = pd.DataFrame(vals, index=row, columns=col)
            if save_format == "csv":
                df.to_csv('..\文件\obj_num.csv')
            elif save_format == "xlsx":
                df.to_excel('..\文件\obj_num.xlsx')
            else:
                print("不支持保存此格式！！！！")

    def del_file(self, path):
        """
        函数功能：删除文件夹下所有文件
        :param path:路径
        :return:
        """
        ls = os.listdir(path)
        for i in ls:
            c_path = os.path.join(path, i)
            if os.path.isdir(c_path):  # 如果是文件夹那么递归调用一下
                self.del_file(c_path)
            else:  # 如果是一个文件那么直接删除
                os.remove(c_path)

    def dataset_balance_val_split(self, val_image_num=60, val_class_num=10):
        """
        函数功能：智能分配训练集和验证集
        :param val_image_num: 验证集数量
        :param val_class_num: 每个类别最低目标数量
        :return:
        """
        # 读取文件列表
        images = os.listdir(self.images_path)
        random.shuffle(images)  # 随机一下文件排序

        # 创建路径
        train_image_path = self.dataset_save_path + "\\train\\images\\"
        train_label_path = self.dataset_save_path + "\\train\\labels\\"
        val_image_path = self.dataset_save_path + "\\val\\images\\"
        val_label_path = self.dataset_save_path + "\\val\\labels\\"
        if not os.path.exists(train_image_path):
            os.makedirs(train_image_path)
        if not os.path.exists(train_label_path):
            os.makedirs(train_label_path)
        if not os.path.exists(val_image_path):
            os.makedirs(val_image_path)
        if not os.path.exists(val_label_path):
            os.makedirs(val_label_path)

        # 清除文件
        self.del_file(train_image_path)
        self.del_file(train_label_path)
        self.del_file(val_image_path)
        self.del_file(val_label_path)

        # 首先随机分配一半的验证集过去
        half_val_image_num = 10
        for img in tqdm(images[:half_val_image_num], desc="初次分配验证集"):
            shutil.copyfile(self.images_path + "\\" + img, val_image_path + img)
            if os.path.exists(self.labels_path + "\\" + img[:-len(self.img_tail[0])] + self.label_tail[0]):
                shutil.copyfile(self.labels_path + "\\" + img[:-len(self.img_tail[0])] + self.label_tail[0],
                                val_label_path + img[:-len(self.img_tail[0])] + self.label_tail[0])
            elif os.path.exists(self.labels_path + "\\" + img[:-len(self.img_tail[1])] + self.label_tail[0]):
                shutil.copyfile(self.labels_path + "\\" + img[:-len(self.img_tail[1])] + self.label_tail[0],
                                val_label_path + img[:-len(self.img_tail[1])] + self.label_tail[0])
            else:
                print("文件后缀异常！！！！")

        # 获取剩下的文件列表
        images = images[half_val_image_num:]

        # 定义记录图片数量的字典
        class_num = copy.deepcopy(self.class_list)
        label_name = ""
        i = 0  # 用于记录是否达到要分配验证集的数量
        val_num = {}
        val_ratio = {}
        N = False  # 标记验证集是否满足条件

        for img in tqdm(images, desc="再次分配验证集"):
            # 获取目标个数
            if i < (val_image_num - half_val_image_num):
                # 获取文件列表
                val_txt = os.listdir(val_label_path)
                # 找到txt文件
                val_txt = self.seek_txt(val_txt)
                val_num, val_ratio = self.object_num(val_txt, val_label_path, self.class_list)

            # 获取路径
            image = os.path.join(self.images_path, img)
            label = ""
            if os.path.exists(self.labels_path + "\\" + img[:-len(self.img_tail[0])] + self.label_tail[0]):
                label_name = img[:-len(self.img_tail[0])] + self.label_tail[0]
                label = os.path.join(self.labels_path, label_name)
            elif os.path.exists(self.labels_path + "\\" + img[:-len(self.img_tail[1])] + self.label_tail[0]):
                label_name = img[:-len(self.img_tail[1])] + self.label_tail[0]
                label = os.path.join(self.labels_path, label_name)
            else:
                print("文件后缀异常！！！！")
            # 将数据记录的字典清空
            for k, v in class_num.items():
                class_num[k] = 0
            self.object_num_once(label, self.class_list, class_num)

            # 检查不满足条件的添加
            if i < (val_image_num - half_val_image_num):
                for k, v in val_num.items():  # 寻找缺少的类别图片
                    if v < val_class_num:
                        if class_num[k] > 0:
                            shutil.copyfile(image, val_image_path + img)
                            shutil.copyfile(label, val_label_path + label_name)
                            i += 1
                            break
                for k, v in val_num.items():  # 判断数据集目标数量是否达标
                    if v >= val_class_num:
                        N = True
                    else:
                        N = False
                        break

                if N:  # 如果验证集已经满足条件，但是验证集数量还不够则随便拷贝补齐
                    shutil.copyfile(image, val_image_path + img)
                    shutil.copyfile(label, val_label_path + label_name)
                    i += 1
                else:
                    if not os.path.exists(val_image_path + img):  # 检查此图片是否已经分配给验证集
                        shutil.copyfile(image, train_image_path + img)
                        shutil.copyfile(label, train_label_path + label_name)
            else:
                # 如果验证集满了，就作为训练集
                shutil.copyfile(image, train_image_path + img)
                shutil.copyfile(label, train_label_path + label_name)

        # 获取文件列表
        train_txt = os.listdir(train_label_path)
        # 找到txt文件
        train_txt = self.seek_txt(train_txt)
        train_num, train_ratio = self.object_num(train_txt, train_label_path, self.class_list)
        # 将数据拼接为一个numpy矩阵
        col = list(val_num)
        row = ["训练集目标数量", "训练集各类别占比", "验证集目标数量", "验证集各类别占比"]
        list1 = list(val_num.values())
        list2 = list(val_ratio.values())
        list3 = list(train_num.values())
        list4 = list(train_ratio.values())
        all_data = np.array([list3, list4, list1, list2])
        # 保存结果
        self.save_result(row, col, all_data.T, save_format="plt_table")


# 测试
if __name__ == '__main__':
    a = dataset_base("../文件/dataset_label.yaml")
    a.dataset_balance_val_split()
