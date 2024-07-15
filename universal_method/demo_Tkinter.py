"""
    Tkinter-python编程
"""


import tkinter as tk
import PyQt5


# 第一个测试程序
def tk_test01():
    window = tk.Tk()
    window.title('my window')
    window.geometry('200x100')

    l = tk.Label(window, text='OMG! this is TK!', bg='green', font=('Arial', 12), width=15, height=2)
    l.pack()

    window.mainloop()  # 循环刷新窗口


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    tk_test01()