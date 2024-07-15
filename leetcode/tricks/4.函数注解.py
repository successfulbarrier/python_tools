#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:leetcode_test
# author:机灵巢穴_WitNest
# datetime:2023/8/14 17:57
# software: PyCharm
from typing import *  # 包含更多注解方式


def func(a: int, b: List[int]):
    ...


#  也可以自定义类
class aa:
    ...


# 注解之后会有提示
def sss(node: aa):
    ...


# 也可以是函数
def func2(x: float) -> int:
    ...


# 注解函数
def func3(f: Callable[[float], int]):
    ...
