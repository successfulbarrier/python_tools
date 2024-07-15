#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:leetcode_test
# author:机灵巢穴_WitNest
# datetime:2023/8/14 20:40
# software: PyCharm

a = eval("10")
print(a)
print(type(a))


b = eval("-10")
print(b)
print(type(b))

c = eval("abs(-10)")
print(c)
print(type(c))


# 一行代码实现n个数相乘
def f(n):
    return eval("*".join([str(i + 1) for i in range(n)]))


print(f(3))

