#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:leetcode_test
# author:机灵巢穴_WitNest
# datetime:2023/8/14 15:55
# software: PyCharm

"""
    方案一
"""


def createArray2D(h, w):
    arr = []
    for y in range(h):
        line = []
        for x in range(w):
            line.append(x)
        arr.append(line)
    return arr


"""
    方案二
"""


def createArray2D_2(h, w):
    return [[i for i in range(w)] for _ in range(h)]


"""
    显示函数
"""


def show(arr):
    print("-----------------")
    for line in arr:
        print(line)
    print("-----------------")


"""
    错误示范
"""


def test1():
    # 好像通过以下方式也可以创建二维数组
    arr = [[1, 2, 3]] * 4
    print(arr)
    # [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]
    # 看似是二维数组实际上在内存中只有一份，这四个[1, 2, 3]公用一份数据
    # 所以第一维可以用乘法，第二维要用推导式


if __name__ == '__main__':
    show(createArray2D_2(10, 20))
    test1()


