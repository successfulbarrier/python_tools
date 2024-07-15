# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   test04.py
# @Time    :   2024/05/21 18:15:27
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   倒立摆角度控制模型

import numpy as np
import matplotlib.pyplot as plt
import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 倒立摆简化模型
class InvertedPendulum:
    def __init__(self, initial_angle=0, initial_angular_velocity=0, length=1.0, mass=1.0, gravity=9.81):
        self.angle = initial_angle
        self.angular_velocity = initial_angular_velocity
        self.length = length
        self.mass = mass
        self.gravity = gravity

    def update(self, force, dt=0.1):
        # 简化的倒立摆动力学模型
        torque = force * self.length
        angular_acceleration = (self.gravity / self.length) * np.sin(self.angle) + torque / (self.mass * self.length**2)
        self.angular_velocity += angular_acceleration * dt
        self.angle += self.angular_velocity * dt

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
    pendulum = InvertedPendulum(initial_angle=np.pi / 8)  # 初始角度30度
    pid = PIDController(kp=40, ki=1, kd=20, setpoint=0)  # 设定值为0，即保持垂直

    def update_setpoint():
        try:
            new_setpoint = float(setpoint_entry.get())
            pid.setpoint = new_setpoint
        except ValueError:
            pass

    root = tk.Tk()
    root.title("倒立摆控制系统")

    fig, ax = plt.subplots()
    ax.set_ylim(-np.pi, np.pi)
    line, = ax.plot([], [], lw=2)
    xdata, ydata = [], []

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    control_frame = tk.Frame(root)
    control_frame.pack(side=tk.BOTTOM)

    tk.Label(control_frame, text="设定值:").pack(side=tk.LEFT)
    setpoint_entry = tk.Entry(control_frame)
    setpoint_entry.pack(side=tk.LEFT)
    setpoint_entry.insert(0, "0")
    tk.Button(control_frame, text="更新设定值", command=update_setpoint).pack(side=tk.LEFT)

    start_time = time.time()
    frame = 0

    def update_plot():
        nonlocal frame
        current_time = time.time()
        elapsed_time = current_time - start_time

        force = pid.compute(pendulum.angle)
        pendulum.update(force)

        xdata.append(elapsed_time)
        ydata.append(pendulum.angle)

        line.set_data(xdata, ydata)
        ax.set_xlim(0, elapsed_time)

        canvas.draw()
        frame += 1
        root.after(100, update_plot)  # 每0.1秒更新一次

    root.after(100, update_plot)
    root.mainloop()

if __name__ == "__main__":
    main()

