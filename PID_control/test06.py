# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   test06.py
# @Time    :   2024/05/21 19:25:11
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   采样时间为0.01的平衡车模型


import numpy as np
import matplotlib.pyplot as plt
import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

class PIDController:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint    # 目标值
        self.integral = 0   # 误差积分
        self.previous_error = 0    # 上一次的误差值

    def compute(self, current_value):
        error = self.setpoint - current_value
        self.integral += error
        derivative = error - self.previous_error
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

def main():
    car = BalanceCar(initial_angle=np.pi / 6)  # 初始角度30度
    angle_pid = PIDController(kp=150, ki=5, kd=40, setpoint=0)  # 角度控制PID
    speed_pid = PIDController(kp=20, ki=0.02, kd=5.0, setpoint=10)  # 速度控制PID

    def update_setpoint():
        try:
            new_setpoint = float(setpoint_entry.get())
            angle_pid.setpoint = new_setpoint
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

        angle_control = angle_pid.compute(car.angle)
        speed_control = speed_pid.compute(car.speed)
        motor_force = angle_control + speed_control

        car.update(motor_force)

        xdata.append(elapsed_time)
        ydata.append(car.angle)

        line.set_data(xdata, ydata)
        ax.set_xlim(0, elapsed_time)

        canvas.draw()
        frame += 1
        root.after(10, update_plot)  # 每0.1秒更新一次

    root.after(10, update_plot)
    root.mainloop()

if __name__ == "__main__":
    main()