# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   test13.py
# @Time    :   2024/06/20 09:54:49
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   新创建的python文件

import cv2
import numpy as np

def detect_defects1(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("无法读取图像")
        return

    # 应用高斯模糊
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # 使用Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 找到轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓
    defect_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(defect_image, contours, -1, (0, 0, 255), 2)

    # 显示结果
    cv2.imshow("Defects", defect_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_defects2(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("无法读取图像")
        return

    # 应用高斯模糊
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # 应用自适应阈值
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 使用形态学操作去除噪声
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 找到轮廓
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓
    defect_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(defect_image, contours, -1, (0, 0, 255), 2)

    # 显示结果
    cv2.imshow("Defects", defect_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_rust(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像")
        return

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义锈蚀的颜色范围
    lower_rust = np.array([0, 50, 50])
    upper_rust = np.array([20, 255, 255])

    # 创建掩膜
    mask = cv2.inRange(hsv, lower_rust, upper_rust)

    # 使用形态学操作去除噪声
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 找到轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓
    rust_image = image.copy()
    cv2.drawContours(rust_image, contours, -1, (0, 0, 255), 2)

    # 显示结果
    cv2.imshow("Rust Detection", rust_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "test/3.jpg"  # 替换为你的图像路径
    detect_rust(image_path)
    # detect_defects2(image_path)
