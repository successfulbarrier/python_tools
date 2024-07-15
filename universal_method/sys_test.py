#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:ultralytics_yolov3
# author:机灵巢穴_WitNest
# datetime:2023/9/9 10:18
# software: PyCharm
"""
    sys模块的一些常见用法
"""

import sys

# 获取Python解释器的版本信息
print(sys.version)

# 获取Python的安装路径
print(sys.executable)

# 获取Python的库路径
print(sys.path)

# 获取Python的编码方式
print(sys.getdefaultencoding())

# 获取Python的编译器版本
print(sys.version_info)

# 获取Python的编译选项
print(sys.flags)

# 获取Python的库路径列表
print(sys.path)

# 获取Python的模块搜索路径列表
print(sys.path_hooks)

# 获取Python的导入模块列表
print(sys.modules)
