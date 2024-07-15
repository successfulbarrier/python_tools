#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:leetcode_test
# author:机灵巢穴_WitNest
# datetime:2023/8/15 18:16
# software: PyCharm

"""
    普通递归算法
"""


def f(n):
    if n == 1:
        return 1
    if n == 2:
        return 1
    else:
        return f(n - 1) + f(n - 2)


def test01():
    for i in range(1, 1000):
        print(i, f(i))


"""
    快速递归，记住之前遍历过的数
"""

from functools import lru_cache


@lru_cache(None)
def f2(n):
    if n == 1:
        return 1
    if n == 2:
        return 1
    else:
        return f(n - 1) + f(n - 2)


def test02():
    for i in range(1, 1000):
        print(i, f2(i))


if __name__ == '__main__':
    test01()
