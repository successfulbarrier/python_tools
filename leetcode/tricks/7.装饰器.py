 #!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:leetcode_test
# author:机灵巢穴_WitNest
# datetime:2023/8/14 21:23
# software: PyCharm


"""
    装饰器，对一个函数的进行外部包装，可以对输入输出参数进行预处理
"""


# 将输入数据和输出数剧
def stradd(oldfunc):
    def newfunc(a: str, b: str):
        a_n = int(a)
        b_n = int(b)
        c = oldfunc(a_n, b_n)
        c_n = str(c)
        return c_n

    return newfunc


# 本来是两个数求和，通过添加装饰器改为两个字符串求和，并返回字符串
@stradd
def add(a: int, b: int):
    return a+b


print(add("10", "20"))
print(type(add("10", "20")))


"""
    通过装饰器来记录函数运行次数
"""


def counter(oldfunc):
    count = 0   # 相当于全局变量

    def newfunc(arr):
        nonlocal count  # 需要声明全局变量
        count += 1
        oldfunc(arr)
        print(f"这个函数{oldfunc.__name__}运行了{count}次")

    return newfunc


@counter
def hello(name: str):
    print(f"hello {name}")


hello("haha")
hello("hehe")


"""
    用于函数报错,出错之后不会导致程序直接崩溃
"""


def shield(oldfunc):
    def newfunc(name: str):
        try:
            oldfunc(name)
        except Exception as e:
            print(f"函数{oldfunc.__name__}出错了", oldfunc, e)

    return newfunc


@shield
def hello2(name: str):
    print(f"hello {name}")


hello2("hehe")


"""
    多次装饰，可以允许无限次装饰
"""


def decorate1(oldfunc):
    def newfunc(name: str):
            print(f"decorate1 hello {name}")
            oldfunc(name)

    return newfunc


def decorate2(oldfunc):
    def newfunc(name: str):
            print(f"decorate2 hello {name}")
            oldfunc(name)

    return newfunc


@decorate2
@decorate1
def hello3(name: str):
    print(f"hello {name}")


hello3("lala")


"""
    带参数的装饰器
"""


def timer1(time_type: str):
    print(time_type)
    if time_type == "min":
        def decorate(oldfunc):
            print("--------------")

            def newfunc(name: str):
                print(f"if hello {name}")
                oldfunc(name)

            return newfunc

    else:
        def decorate(oldfunc):
            print("--------------")

            def newfunc(name: str):
                print(f"else hello {name}")
                oldfunc(name)

            return newfunc


# 教程中可以，这里我的不行，可能版本问题？
@timer1('min')
def hello4(name: str):
    print(f"hello {name}")


hello4("hahe")


# 加快程序运行速度
from functools import lru_cache


@lru_cache(None)
def hello4():
    print("hello")


"""
    类装饰器
"""


class check(object):
    def __init__(self, oldfunc):
        self.oldfunc = oldfunc  # 传入旧函数

    def __call__(self, *args, **kwargs):    # 相当于newfun函数
        self.oldfunc()


@check
def hello4():
    print("hello")