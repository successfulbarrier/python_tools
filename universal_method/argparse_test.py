#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:ultralytics_yolov3
# author:机灵巢穴_WitNest
# datetime:2023/9/9 9:19
# software: PyCharm

"""
    命令行参数传递模块的使用
"""

"""
    argparse模块参数选择
    type 传入参数的类型
    help 帮助信息
    default 默认值
    required 是否为必须参数
    nargs：指定参数接受的值的数量。可以是一个整数，表示接受的值的数量；也可以是'?'或'*'，分别表示接受0个或任意多个值。
    choices：指定参数可以接受的值的可选范围。可以是一个可迭代对象，其中每个元素都是一个字符串，表示一个可选值。列表
    metavar：在帮助信息中显示的参数值示例。
    dest：将参数的值存储到指定的属性或变量中
    action：指定当参数在命令行中出现时要执行的动作。常见的动作包括：
        store：存储参数的值，并将其解析为指定的类型。
        store_const：存储一个常量值作为参数的值。
        store_true 或 store_false：存储布尔值True或False作为参数的值。
        append：将参数的值追加到一个列表中。
        append_const：将一个常量值追加到列表中。
        count：统计参数在命令行中出现的次数。
        help：显示帮助信息并退出程序。
        version：显示版本信息并退出程序。
"""

import argparse

parser = argparse.ArgumentParser(description='姓名')
parser.add_argument('-f', '--family', type=str, help='姓')   # 前两个一个为简化选项，一个为全选项
parser.add_argument('-n', '--name', type=str, required=True, default='', help='名')

# 获取全部参数
args = parser.parse_args()

# 通过对应名称获取参数值
print(args.family+args.name)
