#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:leetcode_test
# author:机灵巢穴_WitNest
# datetime:2023/8/14 20:57
# software: PyCharm

# 两个变量交换数值
a = 1
b = 2
a, b = b, a
print(a, b)
# 可应用于多个变量
c = 3
d = 4
e = 5
c, d, e = d, e, c
print(c, d, e)

# 将元组进行解包,列表也可以
a, *b, c = (1, 2, 3, 4, 5, 6, 7, 8, 9)
print(a)
print(b)
print(c)


# 函数传参
def FUN(*n, **m):
    print(n)
    print(m)


FUN(1, 2, 3, cat=2, trick=3)

