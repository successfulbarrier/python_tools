# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   test09.py
# @Time    :   2024/05/16 17:24:27
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   新创建的python文件

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import os

color_list = ["red","blue","green","yellow"]
# color_list2 = [(240,120,96),(168,184,120),(160,200,160),(56,224,224)]
color_list2 = [(240/255, 120/255, 96/255), 
               (168/255, 184/255, 120/255), 
               (160/255, 200/255, 160/255), 
               (56/255, 224/255, 224/255)]
color_list3 = [(86/255, 129/255, 87/255), 
               (101/255, 135/255, 181/255), 
               (80/255, 81/255, 139/255), 
               (107/255, 33/255, 74/255)]

color_list4 = [(86/255, 129/255, 87/255), 
               (101/255, 135/255, 181/255), 
               (80/255, 81/255, 139/255), 
               (107/255, 33/255, 74/255),
               (240/255, 120/255, 96/255), 
               (168/255, 184/255, 120/255), 
               (160/255, 200/255, 160/255), 
               (56/255, 224/255, 24/255),
               (23/255, 120/255, 96/255), 
               (76/255, 184/255, 120/255), 
               (17/255, 100/255, 160/255), 
               (132/255, 224/255, 224/255),
               (233/255, 23/255, 56/255), 
               (34/255, 243/255, 76/255), 
               (34/255, 23/255, 224/255),
               (127/255, 56/255, 98/255)]

# 读取yolov8中的results中的数据
def read_yolov8_result(file_path):
    data = pd.read_csv(file_path)
    np_data = np.array(data)
    a = np_data[:, 4:8]
    return a

# 绘制一组EMA和没有EMA的曲线
def draw_EMA_line(data):
    # 处理数据
    X = np.linspace(0, data.shape[0], data.shape[0])
    y = data

    # 定义EMA函数
    def exponential_moving_average(data, alpha):
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    # 计算平滑数据
    alpha = 0.1  # 平滑因子
    y_smooth = exponential_moving_average(y, alpha)

    # 可视化
    plt.plot(X, y, color='blue', label='Original Data')
    plt.plot(X, y_smooth, color='red', label='Smoothed Data (EMA)')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Exponential Moving Average')
    plt.legend()
    plt.show()

# 用4个坐标分别绘制四个图像
def draw_EMA_line4(data, title_list=None, alpha = 0.1):
    # 定义EMA函数
    def exponential_moving_average(data, alpha):
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i, title in enumerate(title_list):
        # 处理数据
        X = np.linspace(0, data[i].shape[0], data[i].shape[0])
        y = data[i]
        # 计算平滑数据
        y_smooth = exponential_moving_average(y, alpha)
        # 可视化
        if i == 0:
            axs[0, 0].plot(X, y, color='blue', label='Original Data')
            axs[0, 0].plot(X, y_smooth, color='red', label='Smoothed Data (EMA)')
            axs[0, 0].set_xlabel('X')
            axs[0, 0].set_ylabel('y')
            axs[0, 0].set_title(title)
        elif i == 1:
            axs[0, 1].plot(X, y, color='blue', label='Original Data')
            axs[0, 1].plot(X, y_smooth, color='red', label='Smoothed Data (EMA)')
            axs[0, 1].set_xlabel('X')
            axs[0, 1].set_ylabel('y')
            axs[0, 1].set_title(title)
        elif i == 2:
            axs[1, 0].plot(X, y, color='blue', label='Original Data')
            axs[1, 0].plot(X, y_smooth, color='red', label='Smoothed Data (EMA)')
            axs[1, 0].set_xlabel('X')
            axs[1, 0].set_ylabel('y')
            axs[1, 0].set_title(title)
        elif i == 3:
            axs[1, 1].plot(X, y, color='blue', label='Original Data')
            axs[1, 1].plot(X, y_smooth, color='red', label='Smoothed Data (EMA)')
            axs[1, 1].set_xlabel('X')
            axs[1, 1].set_ylabel('y')
            axs[1, 1].set_title(title)

    plt.legend()
    plt.show()

# 在一个坐标中绘制多条经过EMA算法的曲线
def draw_EMA_line_one_mult(data, title_list=None, alpha = 0.1, title1="test"):
    # 定义EMA函数
    def exponential_moving_average(data, alpha):
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema
    for i, title in enumerate(title_list):
        X = np.linspace(0, data[i].shape[0], data[i].shape[0])
        # 计算平滑数据
        y_smooth = exponential_moving_average(data[i], alpha)
        plt.plot(X, y_smooth, color=color_list4[i], label=title)
        print(title+" : "+str(y_smooth[-1]))

    plt.xlabel('epoch')
    plt.ylabel('map50')
    plt.title(title1)
    plt.legend()
    plt.show()

def draw_line_one_mult(data, title_list=None, alpha = 0.1):
    for i, title in enumerate(title_list):
        X = np.linspace(0, data[i].shape[0], data[i].shape[0])
        # 计算平滑数据
        plt.plot(X, data[i], color=color_list4[i], label=title)
        print(title+" : "+str(np.max(data[i]))) # 打印最大值

    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Exponential Moving Average')
    plt.legend()
    plt.show()

# 自动读取文件夹下的训练结果，并使用draw_EMA_line_one_mult绘制
def auto_read(show_list=None,dim=1, false_name=None, title="test"):
    # dim=0 准确率；dim=1 召回率；dim=2 map50；dim=3 map50-95
    base_folder = "H:\\code\\yolov9_v1\\runs\\train"
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
    list2 = []
    name = []   # 图例
    for subfolder in subfolders:
        # 拼接成全新的路径
        new_path = os.path.join(subfolder, "results.csv")
        if os.path.exists(new_path):
            if show_list == None:
                data = read_yolov8_result(new_path)
                list2.append(np.swapaxes(data,0,1)[dim])
                name.append(os.path.basename(subfolder))
            else:
                if os.path.basename(subfolder) in show_list:
                    data = read_yolov8_result(new_path)
                    list2.append(np.swapaxes(data,0,1)[dim])
                    name.append(os.path.basename(subfolder) if false_name==None else false_name[os.path.basename(subfolder)])
        else:
            print(f"文件 {new_path} 不存在。")
    draw_EMA_line_one_mult(list2, name, title1=title)
    # draw_line_one_mult(list2, name)


if __name__ == "__main__":
    # a = read_yolov8_result("D:\\我的文件\\科研文件\\实验数据\\TS_dataset实验结果\\TS_dataset5\\yolov8n-no-16-300\\results.csv")
    # b = read_yolov8_result("D:\\我的文件\\科研文件\\实验数据\\TS_dataset实验结果\\TS_dataset5\\yolov8n-p-16-300\\results.csv")
    # list2 = [a,b]   # 输入列表
    # out_list2 = []  # 输出列表
    # name = ["on","p"]   # 图例
    # dim = 2 # 数据维度
    # for data in list2:
    #     out_list2.append(np.swapaxes(data,0,1)[dim]) 

    # draw_EMA_line(a[:,0])
    # draw_EMA_line4(np.swapaxes(a,0,1), ["P","R","map50", "map50-95"])
    # draw_EMA_line_one_mult(np.swapaxes(a,0,1), ["P","R","map50", "map50-95"])
    # draw_EMA_line_one_mult(out_list2, name)
    list01 = ["yolov9-c", "yolov9-c-CBAM-G", "yolov9-c-CBAM-ODConv2d"]
    list02 = ["yolov9-c", "yolov9-c_ADown_01", "yolov9-c_ADown_02"]
    list03 = ["yolov9-c", "yolov9-c_lnner_iou_1", "yolov9-c_lnner_iou_1_5", "yolov9-c_lnner_iou_1_5-2"]
    list04 = ["yolov9-c", "yolov9-c_ODConv2d_01", "yolov9-c_ODConv2d_02"]
    list05 = ["yolov9-c", "yolov9-c-Rep_2_hou", "yolov9-c-Rep_2_qian"]
    list06 = ["yolov9-c", "yolov9-c_ODConv2d_lnner_0_5-2", "yolov9-c_ODConv2d_lnner_1_5-2"]
    # 对比实验
    list07 = ["yolov9-c", "yolov8l-no-8-300", "yolov8n-no-16-300", "yolov9-c_ODConv2d_02"]
    list07_name = {"yolov9-c":"ours", "yolov8l-no-8-300":"yolov8l", "yolov8n-no-16-300":"yolov5l", "yolov9-c_ODConv2d_02":"yolov9-c"}
    list07_title = "contrast experiment"
    # 消融实验
    list08 = ["yolov9-c", "yolov9-c_lnner_iou_1", "yolov9-c-CBAM-ODConv2d", "yolov9-c_ODConv2d_02"]
    list08_name = {"yolov9-c":"yolov9-c+CBAM+ODConv+lnner_iou(ours)", "yolov9-c_lnner_iou_1":"yolov9-c+CBAM+ODConv", "yolov9-c-CBAM-ODConv2d":"yolov9-c+ODConv", "yolov9-c_ODConv2d_02":"yolov9-c"}
    list08_title = "ablation experiment"

    auto_read(show_list=list08, false_name=list08_name, title=list08_title)