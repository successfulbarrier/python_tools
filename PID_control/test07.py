# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   test07.py
# @Time    :   2024/05/21 19:31:59
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   使用LQR方法控制平衡车

import numpy as np
import matplotlib.pyplot as plt
import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.linalg import solve_continuous_are


class BalanceCar:
    def __init__(self, initial_angle=0, initial_angular_velocity=0, initial_speed=0, length=1.0, mass=1.0, gravity=9.81):
        self.angle = initial_angle
        self.angular_velocity = initial_angular_velocity
        self.speed = initial_speed
        self.length = length
        self.mass = mass
        self.gravity = gravity

    def update(self, motor_force, dt=0.01):
        # 更复杂的平衡车动力学模型
        torque = motor_force * self.length
        angular_acceleration = (self.gravity / self.length) * np.sin(self.angle) + torque / (self.mass * self.length**2)
        self.angular_velocity += angular_acceleration * dt
        self.angle += self.angular_velocity * dt
        self.speed += motor_force * dt

def lqr(A, B, Q, R):
    # 求解连续时间的Algebraic Riccati方程
    X = solve_continuous_are(A, B, Q, R)
    # 计算LQR增益
    K = np.linalg.inv(R) @ B.T @ X
    return K

def main():
    car = BalanceCar(initial_angle=np.pi / 6)  # 初始角度30度

    # 系统状态空间模型
    A = np.array([[0, 1, 0],
                  [9.81, 0, 0],
                  [0, 0, 0]])
    B = np.array([[0], [1], [1]])

    # LQR权重矩阵
    Q = np.diag([10, 1, 1])
    R = np.array([[1]])

    # 计算LQR增益
    K = lqr(A, B, Q, R)

    def update_setpoint():
        try:
            new_setpoint = float(setpoint_entry.get())
            car.angle = new_setpoint
        except ValueError:
            pass

    root = tk.Tk()
    root.title("平衡车控制系统")

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

        # 状态向量，在实际系统中需要我们通过传感器反馈获取的量
        state = np.array([car.angle, car.angular_velocity, car.speed])
        # 计算控制输入
        motor_force = -K @ state
        motor_force = motor_force.item()  # 将 motor_force 转换为标量

        car.update(motor_force)

        xdata.append(elapsed_time)
        ydata.append(car.angle)

        line.set_data(xdata, ydata)
        ax.set_xlim(0, elapsed_time)

        canvas.draw()
        frame += 1
        root.after(10, update_plot)  # 每0.01秒更新一次

    root.after(10, update_plot)
    root.mainloop()

if __name__ == "__main__":
    main()
