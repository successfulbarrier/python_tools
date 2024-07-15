#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:python_tools
# author:机灵巢穴_WitNest
# datetime:2023/10/28 21:37
# software: PyCharm

"""
    智能数据集分配，保证验证集中各个类别的目标数量不低于设定的最小值
"""

import shutil
import random
import os
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from ADDW import read_classes


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


def object_num_once(file_path, class_list, class_num):
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

    # 遍历计算目标个数 tqdm(file_list, desc="计算目标数量")
    for file in file_list:
        file_path = os.path.join(root_path, file)
        object_num_once(file_path, class_list, class_num)

    # 统计数据
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
            df.to_csv('obj_num.csv')
        elif save_format == "xlsx":
            df.to_excel('obj_num.xlsx')
        else:
            print("不支持保存此格式！！！！")


def del_file(path):
    """
    函数功能：删除文件夹下所有文件
    :param path:路径
    :return:
    """
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):   # 如果是文件夹那么递归调用一下
            del_file(c_path)
        else:                  # 如果是一个文件那么直接删除
            os.remove(c_path)


if __name__ == '__main__':
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description='Process some integers.')

    # 传递带值的参数
    parser.add_argument('--dataset_root', type=str, 
                        default="H:\\LHT\\数据集原始数据\\light_dataset\\new_add_data\\new_dataset\\T_images\\", help='name of the user')
    parser.add_argument('--image_path', type=str, 
                        default="H:\\LHT\\数据集原始数据\\light_dataset\\new_add_data\\new_dataset\\T_images\\", help='name of the user')
    parser.add_argument('--label_path', type=str, 
                        default="H:\\LHT\\数据集原始数据\\light_dataset\\new_add_data\\new_dataset\\T_labels\\", help='name of the user')
    parser.add_argument('--names_path', type=str, 
                        default="H:\\LHT\\数据集原始数据\\light_dataset\\new_add_data\\new_dataset\\T_labels\\", help='name of the user')
    
    # 解析参数
    args = parser.parse_args()

    image_path = args.image_path
    label_path = args.label_path
    # 设定基础参数
    # dataset_root = "H:\\实验数据\\TS_dataset2"
    # image_path = dataset_root + "\\images"  # 图片路径
    # label_path = dataset_root + "\\labels"  # 标注文件路径
    img_tail = [".jpg", ".JPG"]   # 图片后缀
    label_tail = ".txt"     # 标签文件后缀
    dataset_save_path = args.dataset_root  # 分好的数据集保存位置
    val_image_num = 200    # 验证集的数量
    val_class_num = 20     # 验证集每个类别最少要包含的目标数量
    # class_list = {'crack': 0, 'rust': 1, 'broken_lot': 2, 'foreign_matter': 3, 'oil_leak': 4}
    # class_list = {'aeroplane': 0,'bicycle':1,'bird':2,'boat':3,'bottle':4,'bus':5,'car':6,'cat':7,'chair':8,
    #               'cow':9,'diningtable':10,'dog':11,'horse':12,'motorbike':13,'person':14,'pottedplant':15,
    #               'sheep':16,'sofa':17,'train':18,'tvmonitor':19}
    # class_list = {'crack': 0, 'rust': 1, 'broken_lot': 2, 'foreign_matter': 3, 'oil_leak': 4, 'glue_rcolor': 5, 
    #               'WAJR': 6, 'door_open': 7, 'press_plate_on': 8, 'press_plate_off': 9}    
    class_list = read_classes(args.names_path, befor_num=False)
    # 读取文件列表
    images = os.listdir(image_path)
    random.shuffle(images)  # 随机一下文件排序

    # 创建路径
    train_image_path = dataset_save_path + "\\train\\images\\"
    train_label_path = dataset_save_path + "\\train\\labels\\"
    val_image_path = dataset_save_path + "\\val\\images\\"
    val_label_path = dataset_save_path + "\\val\\labels\\"
    if not os.path.exists(train_image_path):
        os.makedirs(train_image_path)
    if not os.path.exists(train_label_path):
        os.makedirs(train_label_path)
    if not os.path.exists(val_image_path):
        os.makedirs(val_image_path)
    if not os.path.exists(val_label_path):
        os.makedirs(val_label_path)

    # 清除文件
    del_file(train_image_path)
    del_file(train_label_path)
    del_file(val_image_path)
    del_file(val_label_path)

    # 首先随机分配一半的验证集过去
    half_val_image_num = 50
    for img in tqdm(images[:half_val_image_num], desc="初次分配验证集"):
        shutil.copyfile(image_path+"\\"+img, val_image_path+img)
        if os.path.exists(label_path+"\\"+img[:-len(img_tail[0])]+label_tail):
            shutil.copyfile(label_path+"\\"+img[:-len(img_tail[0])]+label_tail,
                            val_label_path+img[:-len(img_tail[0])]+label_tail)
        elif os.path.exists(label_path+"\\"+img[:-len(img_tail[1])]+label_tail):
            shutil.copyfile(label_path+"\\"+img[:-len(img_tail[1])]+label_tail,
                            val_label_path+img[:-len(img_tail[1])]+label_tail)
        else:
            print("文件后缀异常！！！！")

    # 获取剩下的文件列表
    images = images[half_val_image_num:]

    # 统计目前验证集各个类别的目标数量
    # class_list = {'light_off_red': 0, 'light_on_red': 1, 'light_off_green': 2, 'light_on_green': 3,
    #               'light_off_yellow': 4, 'light_on_yellow': 5, 'light_off_white': 6, 'light_on_white': 7,
    #               'switch_one_0': 8, 'switch_one_270': 9, 'switch_two_0': 10, 'switch_two_270': 11, 'switch_three_0': 12,
    #               'switch_three_270': 13, 'switch_four_0': 14, 'switch_four_270': 15, 'switch_five_0': 16,
    #               'switch_five_270': 17, 'ya_ban_off': 18, 'ya_ban_on': 19, 'group_red': 20, 'group_green': 21}  # 标签名称

    # 定义记录图片数量的字典
    class_num = copy.deepcopy(class_list)
    label_name = ""
    i = 0  # 用于记录是否达到要分配验证集的数量
    val_num = {}
    val_ratio = {}
    N = False    # 标记验证集是否满足条件

    for img in tqdm(images, desc="再次分配验证集"):
        # 获取目标个数
        if i < (val_image_num-half_val_image_num):
            # 获取文件列表
            val_txt = os.listdir(val_label_path)
            # 找到txt文件
            val_txt = seek_txt(val_txt)
            val_num, val_ratio = object_num(val_txt, val_label_path, class_list)

        # 获取路径
        image = os.path.join(image_path, img)
        if os.path.exists(label_path+"\\"+img[:-len(img_tail[0])]+label_tail):
            label_name = img[:-len(img_tail[0])]+label_tail
            label = os.path.join(label_path, label_name)
        elif os.path.exists(label_path+"\\"+img[:-len(img_tail[1])]+label_tail):
            label_name = img[:-len(img_tail[1])] + label_tail
            label = os.path.join(label_path, label_name)
        else:
            print("文件后缀异常！！！！")
        # 将数据记录的字典清空
        for k, v in class_num.items():
            class_num[k] = 0
        object_num_once(label, class_list, class_num)

        # 检查不满足条件的添加
        if i < (val_image_num-half_val_image_num):
            for k, v in val_num.items():    # 寻找缺少的类别图片
                if v < val_class_num:
                    if class_num[k] > 0:
                        shutil.copyfile(image, val_image_path + img)
                        shutil.copyfile(label, val_label_path + label_name)
                        i += 1
                        break
            for k, v in val_num.items():    # 判断数据集目标数量是否达标
                if v >= val_class_num:
                    N = True
                else:
                    N = False
                    break

            if N:   # 如果验证集已经满足条件，但是验证集数量还不够则随便拷贝补齐
                shutil.copyfile(image, val_image_path + img)
                shutil.copyfile(label, val_label_path + label_name)
                i += 1
            else:
                if not os.path.exists(val_image_path + img):    # 检查此图片是否已经分配给验证集
                    shutil.copyfile(image, train_image_path + img)
                    shutil.copyfile(label, train_label_path + label_name)
        else:
            # 如果验证集满了，就作为训练集
            shutil.copyfile(image, train_image_path + img)
            shutil.copyfile(label, train_label_path + label_name)

    # 将数据拼接为一个numpy矩阵
    col = list(val_num)
    row = ["数据集目标数量", "数据集各类别占比"]
    list1 = list(val_num.values())
    list2 = list(val_ratio.values())
    all_data = np.array([list1, list2])
    # 保存结果
    save_result(row, col, all_data.T, save_format="csv")

