# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   test02.py
# @Time    :   2024/05/21 16:21:56
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   使用python模拟单容水箱的控制过程

"""
1. WaterTank 类：用于模拟水箱系统。update 方法根据阀门开度更新水箱水位。
2. PIDController 类：实现PID控制算法。compute 方法根据当前水位计算阀门开度。
3. main 函数：初始化水箱和PID控制器，并使用 matplotlib 动画显示水位变化。
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# 单容水箱模型
class WaterTank:
    def __init__(self, initial_level=0, inflow_rate=1, outflow_rate=0.5, max_level=100):
        self.level = initial_level
        self.inflow_rate = inflow_rate
        self.outflow_rate = outflow_rate
        self.max_level = max_level

    def update(self, valve_opening):
        inflow = self.inflow_rate * valve_opening
        outflow = self.outflow_rate
        self.level += inflow - outflow
        self.level = max(0, min(self.level, self.max_level))  # 限制水位在0到max_level之间

class PIDController:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0
        self.previous_error = 0

    def compute(self, current_value):
        error = self.setpoint - current_value
        self.integral += error
        derivative = error - self.previous_error
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

def main():
    tank = WaterTank(initial_level=30)
    pid = PIDController(kp=1.0, ki=0.01, kd=0.5, setpoint=70)

    fig, ax = plt.subplots()
    ax.set_ylim(0, 100)
    line, = ax.plot([], [], lw=2)
    xdata, ydata = [], []

    start_time = time.time()
    frame = 0

    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time

        valve_opening = pid.compute(tank.level)
        valve_opening = max(0, min(valve_opening, 1))  # 限制阀门开度在0到1之间
        tank.update(valve_opening)

        xdata.append(elapsed_time)
        ydata.append(tank.level)

        line.set_data(xdata, ydata)
        ax.set_xlim(0, elapsed_time)

        plt.pause(0.1)  # 更新图表

        frame += 1
        time.sleep(0.1)  # 模拟采样周期

if __name__ == "__main__":
    main()
