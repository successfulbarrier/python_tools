#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:leetcode_test
# author:机灵巢穴_WitNest
# datetime:2023/8/14 16:37
# software: PyCharm

"""
    推导式，以下四种python类型都有推导式
    arr = []    列表
    t = ()      元组
    dit = {}    字典
    s = set()   集合
"""


def test1():
    # 通过for循环只保留偶数
    arr1 = []
    for i in range(50):
        if i % 2 == 0:
            arr1.append(i)
        else:
            arr1.append(0)
    print(arr1)

    # 将带有判断的for循环简化为推导式
    arr2 = [i if i % 2 == 0 else 0 for i in range(50)]
    print(arr2)


# 在生成二维数组中也有使用推导式，列表、元组和集合用法相同，字典一般不用
if __name__ == '__main__':
    test1()

