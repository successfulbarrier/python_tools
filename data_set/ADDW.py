# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   ADDW.py
# @Time    :   2024/06/18 18:34:00
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   数据集处理常用函数，个人库


# 读取类别配置文件class.names为字典格式

def read_classes(file_path, befor_num=True):
    names = {}
    num = 0
    with open(file_path, 'r') as f:
        while True:
            line = f.readline()
            if line == '':  # 如果文件结束，结束循环
                break
            names[num] = line.rstrip("\n")  # 改进读取方式
            num += 1
    if befor_num == False:
        names = {v: k for k, v in names.items()}
    # print(names)
    print("-->read names<--")
    return names


if __name__ == '__main__':
    read_classes("H:\\LHT\\数据集原始数据\\light_dataset\\classes.names", befor_num=False)