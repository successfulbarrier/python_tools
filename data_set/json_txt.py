#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:ultralytics_yolov3
# author:机灵巢穴_WitNest
# datetime:2023/9/22 20:47
# software: PyCharm

"""
    将json转txt
"""

import json
import os
from tqdm import tqdm
import argparse
from ADDW import read_classes

# name2id = {'light_off_red': 0, 'light_on_red': 1, 'light_off_green': 2, 'light_on_green': 3, 'light_off_yellow': 4, 'light_on_yellow': 5, 'light_off_white': 6, 'light_on_white': 7, 'switch_one_0': 8, 'switch_one_270': 9, 'switch_two_0': 10, 'switch_two_270': 11, 'switch_three_0': 12, 'switch_three_270': 13, 'switch_four_0': 14, 'switch_four_270': 15, 'switch_five_0': 16, 'switch_five_270': 17, 'ya_ban_off': 18, 'ya_ban_on': 19, 'group_red': 20, 'group_green': 21}  # 标签名称
# name2id = {'crack': 0, 'rust': 1, 'broken_lot': 2, 'foreign_matter': 3, 'oil_leak': 4, 'glue_rcolor': 5, 'WAJR': 6, 'door_open': 7, 'press_plate_on': 8, 'press_plate_off': 9}

# root_path = "H:\\LHT\\数据集原始数据\\light_dataset\\new_add_data\\new_dataset"
# json_path = root_path + "\\T_labels\\"
# txt_path = root_path + "\\labels\\"


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


def decode_json(json_floder_path, json_name, txt_path, name2id):
    # 如果目录不存在就创建目录
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)

    txt_name = txt_path + json_name[0:-5] + '.txt'
    # 存放txt的绝对路径
    txt_file = open(txt_name, 'w')

    json_path = os.path.join(json_floder_path, json_name)
    data = json.load(open(json_path, 'r', encoding='gb2312', errors='ignore'))

    img_w = data['imageWidth']
    img_h = data['imageHeight']

    for i in data['shapes']:

        label_name = i['label']
        if (i['shape_type'] == 'rectangle'):
            x1 = int(i['points'][0][0])
            y1 = int(i['points'][0][1])
            x2 = int(i['points'][1][0])
            y2 = int(i['points'][1][1])

            bb = (x1, y1, x2, y2)
            bbox = convert((img_w, img_h), bb)
            txt_file.write(str(name2id[label_name]) + " " + " ".join([str(a) for a in bbox]) + '\n')


if __name__ == "__main__":
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description='Process some integers.')

    # 传递带值的参数
    parser.add_argument('--json_path', type=str, 
                        default="H:\\LHT\\数据集原始数据\\light_dataset\\new_add_data\\new_dataset\\T_images\\", help='name of the user')
    parser.add_argument('--txt_path', type=str, 
                        default="H:\\LHT\\数据集原始数据\\light_dataset\\new_add_data\\new_dataset\\T_labels\\", help='name of the user')
    parser.add_argument('--names_path', type=str, 
                        default="H:\\LHT\\数据集原始数据\\light_dataset\\new_add_data\\new_dataset\\T_labels\\", help='name of the user')
    
    # 解析参数
    args = parser.parse_args()
    # 存放json的文件夹的绝对路径
    json_names = os.listdir(args.json_path)
    txt_path = args.txt_path
    json_path = args.json_path
    name2id = read_classes(args.names_path, befor_num=False)

    for json_name in tqdm(json_names):
        decode_json(json_path, json_name, txt_path, name2id)

