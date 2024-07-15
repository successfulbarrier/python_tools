# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   test04.py
# @Time    :   2024/05/21 15:50:20
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   在K210上使用python推理模型文件，将模型拷贝到sd卡即可

# 转换模型
# 安装nncase
# pip install nncase

# 转换ONNX模型为Kmodel
# nncase compile --input-model path/to/your/model.onnx --output-model path/to/your/model.kmodel --target k210

# 在开发板上运行的推理代码
import sensor
import image
import lcd
import KPU as kpu
import os
import sys

# 初始化LCD
lcd.init()

# 初始化摄像头
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.run(1)

# 挂载SD卡
if not os.path.exists('/sd'):
    os.mkdir('/sd')
if not os.mountsd():
    print("SD卡挂载失败")
    sys.exit()

# 加载Kmodel
task = kpu.load("/sd/model.kmodel")

while True:
    # 捕获图像
    img = sensor.snapshot()

    # 运行推理
    fmap = kpu.forward(task, img)

    # 获取输出结果
    plist = fmap[:]
    pmax = max(plist)
    max_index = plist.index(pmax)

    # 显示结果
    lcd.display(img)
    print("推理结果: 类别 %d, 概率 %f" % (max_index, pmax))

# 释放资源
kpu.deinit(task)



