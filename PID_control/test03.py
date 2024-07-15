# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   test03.py
# @Time    :   2024/05/21 16:54:26
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   使用python模拟单容水箱的控制过程,并动态设计水位

import numpy as np
import matplotlib.pyplot as plt
import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
    tank = WaterTank(initial_level=50)
    pid = PIDController(kp=2.0, ki=0.03, kd=0.0, setpoint=70)

    def update_setpoint():
        try:
            new_setpoint = float(setpoint_entry.get())
            pid.setpoint = new_setpoint
        except ValueError:
            pass

    root = tk.Tk()
    root.title("水箱控制系统")

    fig, ax = plt.subplots()
    ax.set_ylim(0, 100)
    line, = ax.plot([], [], lw=2)
    xdata, ydata = [], []

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    control_frame = tk.Frame(root)
    control_frame.pack(side=tk.BOTTOM)

    tk.Label(control_frame, text="设定值:").pack(side=tk.LEFT)
    setpoint_entry = tk.Entry(control_frame)
    setpoint_entry.pack(side=tk.LEFT)
    setpoint_entry.insert(0, "70")
    tk.Button(control_frame, text="更新设定值", command=update_setpoint).pack(side=tk.LEFT)

    start_time = time.time()
    frame = 0

    def update_plot():
        nonlocal frame
        current_time = time.time()
        elapsed_time = current_time - start_time

        valve_opening = pid.compute(tank.level)
        valve_opening = max(0, min(valve_opening, 1))  # 限制阀门开度在0到1之间
        tank.update(valve_opening)

        xdata.append(elapsed_time)
        ydata.append(tank.level)

        line.set_data(xdata, ydata)
        ax.set_xlim(0, elapsed_time)

        canvas.draw()
        frame += 1
        root.after(100, update_plot)  # 每0.1秒更新一次

    root.after(100, update_plot)
    root.mainloop()

if __name__ == "__main__":
    main()