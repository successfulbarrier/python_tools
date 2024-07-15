#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:leetcode_test
# author:机灵巢穴_WitNest
# datetime:2023/8/16 17:36
# software: PyCharm

"""
    传统方法
"""

s = "🐕💩💩🐕💩🐕🐕💩💩🐕💩🐕💩🐕💩🐕🐕💩🐕💩💩💩🐕💩🐕💩"

dic = {}
for char in s:
    if char not in dic:
        dic[char] = 1
    else:
        dic[char] += 1

print(dic)


"""
    使用库,进行计数
    执行速度也更快，推荐使用
"""

from collections import Counter

s = "🐕💩💩🐕💩🐕🐕💩💩🐕💩🐕💩🐕💩🐕🐕💩🐕💩💩💩🐕💩🐕💩"
c = Counter(s)
print(c)
