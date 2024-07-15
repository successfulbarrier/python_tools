# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   test01.py
# @Time    :   2024/05/21 15:57:10
# @Author  :   机灵巢穴_WitNest 
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   使用 matplotlib 和 tkinter 来实现一个带有动态曲线的界面。

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import threading
import time


class DynamicPlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("动态曲线显示")

        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_ylim(-100, 100)
        self.ax.set_xlim(0, 100)
        self.ax.grid()

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.ani = animation.FuncAnimation(self.fig, self.update_plot, self.data_gen, blit=True, interval=100, repeat=False)

        # 启动数据处理线程
        self.data_thread = threading.Thread(target=self.process_data)
        self.data_thread.daemon = True
        self.data_thread.start()

    def data_gen(self):
        x = np.linspace(0, 100, 100)
        while True:
            y = np.random.uniform(-100, 100, 100)
            yield x, y

    def update_plot(self, data):
        x, y = data
        self.line.set_data(x, y)
        return self.line,

    def process_data(self):
        while True:
            # 在这里处理你的数据
            print("处理数据中...")
            time.sleep(1)  # 模拟数据处理时间
            

if __name__ == "__main__":
    root = tk.Tk()
    app = DynamicPlotApp(root)
    root.mainloop()
