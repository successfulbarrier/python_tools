#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:ultralytics_yolov3
# author:机灵巢穴_WitNest
# datetime:2023/9/9 9:58
# software: PyCharm

"""
    os模块的一些常见用法
    pathlib是os的升级版
"""
import os
from pathlib import Path

# 获取当前工作目录
current_dir = os.getcwd()

# 改变当前工作目录
os.chdir('/path/to/new/directory')

# 列出指定目录下的文件和文件夹
files = os.listdir('/path/to/directory')

# 创建新目录
os.mkdir('/path/to/new/directory')

# 删除文件或目录
os.remove('/path/to/file')
os.rmdir('/path/to/directory')

# 重命名文件或目录
os.rename('/path/to/old_name', '/path/to/new_name')

# 检查路径是否存在
path_exists = os.path.exists('/path/to/directory')

# 获取文件大小
file_size = os.path.getsize('/path/to/file')

# 判断给定路径是否为文件
is_file = os.path.isfile('/path/to/file')

# 判断给定路径是否为目录
is_directory = os.path.isdir('/path/to/directory')

# 获取系统环境变量
print(os.environ)

# 获取当前进程ID
print(os.getpid())


"""
    获取当前目录下的文件列表排除目录
"""

# 获取当前目录下的所有文件和目录
all_list = os.listdir('..')

# 存储文件名的列表
file_list = []

# 遍历所有文件和目录
for file in all_list:
    # 判断是否为文件, 也可判断是否为目录 os.path.isdir()
    if os.path.isfile(file):
        # 将文件名添加到列表中
        file_list.append(file)

# 打印文件列表
print(file_list)


"""
    Python的pathlib模块提供了一种面向对象的方式来处理文件系统路径。它的主要优点是可以自动处理不同操作系统的路径分隔符，
    并且可以方便地操作路径的各个部分（如目录、文件名等）。
"""
# 创建路径对象
p = Path('/usr/local/bin')

# 获取路径的各个部分
print(p.parts)  # 输出 ['', 'usr', 'local', 'bin']

# 判断路径是否存在
print(p.exists())   # True

# 遍历目录下的所有文件和子目录
for item in p.iterdir():
    print(item)

# 拼接路径
print(p1 / p2) # 输出 /usr/local/bin

# 改变路径的工作目录
p.chdir('/usr/local/share')
print(p) # 输出 /usr/local/share/bin

# 获取文件的大小
print(p.stat().st_size) # 输出文件大小，单位为字节
