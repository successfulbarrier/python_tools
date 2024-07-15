#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:leetcode_test
# author:机灵巢穴_WitNest
# datetime:2023/8/14 17:13
# software: PyCharm

# 复数的定义与计算
v1 = 1 + 1j
v2 = 2 + 2j
print(v1 + v2)  # 方便的求向量相加
print(abs(v1))  # 方便的求到原地的距离

# 通过函数创建复数
v = complex(1, 1)
print(v)
print(v.real)   # 实部
print(v.imag)   # 虚部

