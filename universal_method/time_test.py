#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:ultralytics_yolov3
# author:机灵巢穴_WitNest
# datetime:2023/9/9 9:44
# software: PyCharm
"""
    time模块的一些常见用法
    time库的升级版本
"""

import time
import datetime

# 1.获取当前时间戳：
timestamp = time.time()
print(timestamp)

# 2.将时间戳转换为本地时间：
local_time = time.localtime(timestamp)
print(local_time)

# 3.将本地时间转换为格式化的时间字符串：
formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
print(formatted_time)

# 4.获取当前日期：
date = time.strftime("%Y-%m-%d", local_time)
print(date)

# 5.获取当前时间：
current_time = time.strftime("%H:%M:%S", local_time)
print(current_time)

# 延迟执行代码：
time.sleep(1)  # seconds为要延迟的秒数

# 获取程序运行的CPU时间：
cpu_time = time.process_time()
print(cpu_time)


# 同时执行多个代码块，并测量它们的执行时间：

def func1():
    # code block 1 to measure execution time for func1
    pass


def func2():
    # code block 2 to measure execution time for func2
    pass


start_time = time.perf_counter()
func1()
func2()
end_time = time.perf_counter()
execution_time = end_time - start_time
print("Execution time:", execution_time)


"""
    datetime
"""


now = datetime.datetime.now()
print(now)


now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))


start_time = datetime.datetime(2022, 1, 1, 0, 0, 0)
end_time = datetime.datetime(2022, 1, 2, 0, 0, 0)
delta = end_time - start_time
print(delta)


time1 = datetime.datetime(2022, 1, 1, 0, 0, 0)
time2 = datetime.datetime(2022, 1, 2, 0, 0, 0)
if time1 < time2:
    print("time1 is earlier than time2")
else:
    print("time1 is later than or equal to time2")
