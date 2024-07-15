#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:python_tools
# author:机灵巢穴_WitNest
# datetime:2023/10/21 10:18
# software: PyCharm

import onnxruntime as rt    # 如果要使用GUP推理安装GPU版本即可 onnxruntime-gpu
import numpy as np
import cv2
import random


def nms(pred, conf_thres, iou_thres):
    """
    函数功能：非极大值抑制
    :param pred: 预测结果
    :param conf_thres:conf阈值
    :param iou_thres:iou阈值
    :return:
    """
    conf = pred[..., 4] > conf_thres
    box = pred[conf == True]
    cls_conf = box[..., 5:]
    cls = []
    for i in range(len(cls_conf)):
        cls.append(int(np.argmax(cls_conf[i])))
    total_cls = list(set(cls))
    output_box = []
    for i in range(len(total_cls)):
        clss = total_cls[i]
        cls_box = []
        for j in range(len(cls)):
            if cls[j] == clss:
                box[j][5] = clss
                cls_box.append(box[j][:6])
        cls_box = np.array(cls_box)
        box_conf = cls_box[..., 4]
        box_conf_sort = np.argsort(box_conf)
        max_conf_box = cls_box[box_conf_sort[len(box_conf) - 1]]
        output_box.append(max_conf_box)
        cls_box = np.delete(cls_box, 0, 0)
        while len(cls_box) > 0:
            max_conf_box = output_box[len(output_box) - 1]
            del_index = []
            for j in range(len(cls_box)):
                current_box = cls_box[j]
                interArea = getInter(max_conf_box, current_box)
                iou = getIou(max_conf_box, current_box, interArea)
                if iou > iou_thres:
                    del_index.append(j)
            cls_box = np.delete(cls_box, del_index, 0)
            if len(cls_box) > 0:
                output_box.append(cls_box[0])
                cls_box = np.delete(cls_box, 0, 0)
    return output_box


def getIou(box1, box2, inter_area):
    """
    函数功能：计算iou
    :param box1:
    :param box2:
    :param inter_area:交集
    :return:
    """
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - inter_area
    iou = inter_area / union
    return iou


def getInter(box1, box2):
    """
    函数功能：计算两个框的交集
    :param box1:
    :param box2:
    :return:返回两个框的交集
    """
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2, \
                                         box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[0] - box2[2] / 2, box2[1] - box1[3] / 2, \
                                         box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
    if box1_x1 > box2_x2 or box1_x2 < box2_x1:
        return 0
    if box1_y1 > box2_y2 or box1_y2 < box2_y1:
        return 0
    x_list = [box1_x1, box1_x2, box2_x1, box2_x2]
    x_list = np.sort(x_list)
    x_inter = x_list[2] - x_list[1]
    y_list = [box1_y1, box1_y2, box2_y1, box2_y2]
    y_list = np.sort(y_list)
    y_inter = y_list[2] - y_list[1]
    inter = x_inter * y_inter
    return inter


def draw(img, xscale, yscale, pred, class_name, colour_list):
    """
    函数功能：将预测结果绘制在图片上
    :param img:图片
    :param xscale:x缩放尺度
    :param yscale:y缩放尺度
    :param pred:预测结果
    :param class_name:类别列表
    :param colour_list:类别对应的颜色列表
    :return:绘制好的图片
    """
    img_ = img.copy()
    if len(pred):
        for detect in pred:
            detect = [int((detect[0] - detect[2] / 2) * xscale), int((detect[1] - detect[3] / 2) * yscale),
                      int((detect[0] + detect[2] / 2) * xscale), int((detect[1] + detect[3] / 2) * yscale),
                      float(detect[4]), int(detect[5])]
            img_ = cv2.rectangle(img, (detect[0], detect[1]), (detect[2], detect[3]), colour_list[detect[-1]], 3)
            cv2.putText(img_, class_name[detect[-1]], (detect[0], detect[1]), cv2.FONT_HERSHEY_COMPLEX, 2.0, colour_list[detect[-1]], 5)
    return img_


def class_colour(class_name):
    """
    函数功能：随机生成各个类别对应的颜色
    :param class_name: 类别名称，字典
    :return: 返回随机生成的各个类别的颜色列表
    """
    colour_list = {}
    for k, v in class_name.items():
        (r, g, b) = random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)
        colour_list.update({k: (r, g, b)})
    return colour_list


if __name__ == '__main__':
    # 参数加载
    height, width = 640, 640
    class_name = {'light_off-red': 0, 'light_on-red': 1, 'light_off-green': 2, 'light_on-green': 3, 'light_off-yellow': 4,
                  'light_on-yellow': 5, 'light_off-white': 6, 'light_on-white': 7, 'local': 8, 'on': 9, 'trip': 10,
                  'disjunctor_on': 11, 'off': 12, 'light_off-black': 13}  # 标签名称
    class_name = {v: k for k, v in class_name.items()}
    colour_list = class_colour(class_name)
    image = input("请输入图片路径：")
    img0 = cv2.imread(image)
    # 数据预处理
    x_scale = img0.shape[1] / width
    y_scale = img0.shape[0] / height
    xy_scale = img0.shape[1]/img0.shape[0]
    img = img0 / 255.
    img = cv2.resize(img, (width, height))
    img = np.transpose(img, (2, 0, 1))
    data = np.expand_dims(img, axis=0)
    print("推理使用设备："+rt.get_device())
    # 模型预测
    sess = rt.InferenceSession('best.onnx')     # 加载模型
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred = sess.run([label_name], {input_name: data.astype(np.float32)})[0]     # 传入数据预测结果
    pred = np.squeeze(pred)     # 删除多余的维度
    pred = np.transpose(pred, (1, 0))   # 交换维度
    pred_class = pred[..., 4:]      # 取出类别概率
    pred_conf = np.max(pred_class, axis=-1)     # 计算概率最大的那个类别
    pred = np.insert(pred, 4, pred_conf, axis=-1)   # 插入一个置信度维度
    # 结果处理
    result = nms(pred, 0.3, 0.45)   # 进行非极大值抑制
    ret_img = draw(img0, x_scale, y_scale, result, class_name, colour_list)     # 绘制检测结果
    ret_img = ret_img[:, :, ::-1]   # 得到正常可以显示的图片
    ret_img = cv2.resize(ret_img, (640, int(640/xy_scale)), interpolation=cv2.INTER_LINEAR)     # 调节图片大小
    # 结果显示
    cv2.imshow("result_image", ret_img)     # 显示图片
    cv2.waitKey(0)     # 等待按键推出
    cv2.destroyAllWindows()

