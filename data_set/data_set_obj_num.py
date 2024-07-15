#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:python_tools
# author:机灵巢穴_WitNest
# datetime:2023/10/14 21:47
# software: PyCharm

"""
    计算数据集中各个类别目标的数量,用于评估各个类别样本是否平衡
"""

import os
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def seek_txt(file_list):
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


def object_num(file_list, root_path, class_list):
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

    # 遍历计算目标个数
    for file in tqdm(file_list):
        file_path = os.path.join(root_path, file)
        line = "1"
        with open(file_path, 'r') as f:
            while True:
                num = 0
                line = f.readline()
                if line == '':      # 如果文件结束，结束循环
                    break
                for s in line:      # 寻找第一个数的真实长度
                    if s == ' ':
                        break
                    num += 1

                object_class = int(line[0:num])
                for k, v in class_list.items():
                    if class_list[k] == object_class:
                        class_num[k] += 1
    all_num = 0
    for k, v in class_num.items():
        all_num += class_num[k]
    for k, v in class_num.items():
        class_ratio[k] = round(class_num[k]/all_num, 3) # 保留三位小数

    return class_num, class_ratio   # 数量，占比


def plt_table(col, row, vals):
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
    plt.figure(figsize=(30, 60))    # 设置整幅图像大小
    tab = plt.table(cellText=vals,
                    colLabels=col,
                    rowLabels=row,
                    loc='center',
                    cellLoc='center',
                    rowLoc='center')
    tab.scale(0.8, 10)      # 设置格子比例
    tab.set_fontsize(60)    # 设置字体大小
    plt.axis('off')
    plt.gcf().subplots_adjust(left=0.4, top=0.6, bottom=0.4, right=0.6)
    plt.show()


def save_result(col, row, vals, save_format="csv"):
    """
    # 函数功能保存统计的结果
    :param save_format: 保存的格式
    :param col: 列名
    :param row: 行名
    :param vals: numpy列表值
    :return:
    """
    if save_format == "plt_table":
        plt_table(col, row, vals)
    else:
        df = pd.DataFrame(vals, index=row, columns=col)
        if save_format == "csv":
            df.to_csv('.\文件\obj_num.csv')
        elif save_format == "xlsx":
            df.to_excel('.\文件\obj_num.xlsx')
        else:
            print("不支持保存此格式！！！！")


def train_val_run():
    # 训练集和验证集的txt标签路径
    train_path = "H:/code/datasets/TS_dataset4/train/labels"
    val_path = "H:/code/datasets/TS_dataset4/val/labels"
    # class_list = {'light_off_red': 0, 'light_on_red': 1, 'light_off_green': 2, 'light_on_green': 3,
    #               'light_off_yellow': 4,
    #               'light_on_yellow': 5, 'light_off_white': 6, 'light_on_white': 7, 'switch_one_0': 8,
    #               'switch_one_270': 9,
    #               'switch_two_0': 10, 'switch_two_270': 11, 'switch_three_0': 12, 'switch_three_270': 13,
    #               'switch_four_0': 14,
    #               'switch_four_270': 15, 'switch_five_0': 16, 'switch_five_270': 17, 'ya_ban_off': 18, 'ya_ban_on': 19,
    #               'group_red': 20, 'group_green': 21}  # 标签名称
    # class_list = {'crack': 0, 'rust': 1, 'broken_lot': 2, 'foreign_matter': 3, 'oil_leak': 4}
    
    # class_list = {'aeroplane': 0,'bicycle':1,'bird':2,'boat':3,'bottle':4,'bus':5,'car':6,'cat':7,'chair':8,
    #               'cow':9,'diningtable':10,'dog':11,'horse':12,'motorbike':13,'person':14,'pottedplant':15,
    #               'sheep':16,'sofa':17,'train':18,'tvmonitor':19}
    
    class_list = {'crack': 0, 'rust': 1, 'broken_lot': 2, 'foreign_matter': 3, 'oil_leak': 4, 'glue_rcolor': 5, 
                  'WAJR': 6, 'door_open': 7, 'press_plate_on': 8, 'press_plate_off': 9} 
    
    # 获取文件列表
    train_txt = os.listdir(train_path)
    val_txt = os.listdir(val_path)

    # 找到txt文件
    train_txt = seek_txt(train_txt)
    val_txt = seek_txt(val_txt)

    # 获取目标个数
    train_num, train_ratio = object_num(train_txt, train_path, class_list)
    val_num, val_ratio = object_num(val_txt, val_path, class_list)

    # 将数据拼接为一个numpy矩阵
    col = list(train_num)
    row = ["train_num", "train_ratio", "val_num", "val_ratio"]
    list1 = list(train_num.values())
    list2 = list(train_ratio.values())
    list3 = list(val_num.values())
    list4 = list(val_ratio.values())
    all_data = np.array([list1, list2, list3, list4])

    # 保存结果
    save_result(row, col, all_data.T, save_format="csv")


def  all_run():
    # 训练集和验证集的txt标签路径
    train_path = "H:\\code\\datasets\\VOC\\VOC2012\\val\\labels"
    # class_list = {'light_off_red': 0, 'light_on_red': 1, 'light_off_green': 2, 'light_on_green': 3,
    #               'light_off_yellow': 4,
    #               'light_on_yellow': 5, 'light_off_white': 6, 'light_on_white': 7, 'switch_one_0': 8,
    #               'switch_one_270': 9,
    #               'switch_two_0': 10, 'switch_two_270': 11, 'switch_three_0': 12, 'switch_three_270': 13,
    #               'switch_four_0': 14,
    #               'switch_four_270': 15, 'switch_five_0': 16, 'switch_five_270': 17, 'ya_ban_off': 18, 'ya_ban_on': 19,
    #               'group_red': 20, 'group_green': 21}  # 标签名称
    
    # class_list = {'aeroplane': 0,'bicycle':1,'bird':2,'boat':3,'bottle':4,'bus':5,'car':6,'cat':7,'chair':8,
    #               'cow':9,'diningtable':10,'dog':11,'horse':12,'motorbike':13,'person':14,'pottedplant':15,
    #               'sheep':16,'sofa':17,'train':18,'tvmonitor':19}
    
    class_list = {'crack': 0, 'rust': 1, 'broken_lot': 2, 'foreign_matter': 3, 'oil_leak': 4, 'glue_rcolor': 5, 
                  'WAJR': 6, 'door_open': 7, 'press_plate_on': 8, 'press_plate_off': 9} 
    # 获取文件列表
    train_txt = os.listdir(train_path)

    # 找到txt文件
    train_txt = seek_txt(train_txt)

    # 获取目标个数
    train_num, train_ratio = object_num(train_txt, train_path, class_list)

    # 将数据拼接为一个numpy矩阵
    col = list(train_num)
    row = ["数据集目标数量", "数据集各类别占比"]
    list1 = list(train_num.values())
    list2 = list(train_ratio.values())
    all_data = np.array([list1, list2])

    # 保存结果
    save_result(row, col, all_data.T, save_format="csv")


if __name__ == '__main__':
    train_val_run()

