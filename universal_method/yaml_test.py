#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:ultralytics_yolov3
# author:机灵巢穴_WitNest
# datetime:2023/9/8 21:49
# software: PyCharm
"""
    yaml文件数据读取
    支持的数据类型：
        字典、列表、字符串、布尔值、整数、浮点数、Null、时间等
    基本语法规则：
        1、大小写敏感
        2、使用缩进表示层级关系
        3、相同层级的元素左侧对齐
        4、键值对用冒号 “:” 结构表示，冒号与值之间需用空格分隔
        5、数组前加有 “-” 符号，符号与值之间需用空格分隔
        6、None值可用null 和 ~ 表示
        7、多组数据之间使用3横杠—分割
        8、# 表示注释，但不能在一段代码的行末尾加 #注释，否则会报错
"""
import yaml


# 读取一组数据
def test1():
    with open('../文件/yaml_test.yaml', 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)

    print(result, type(result))
    print(result['os'], type(result['os']))
    print(result['osVersion'], type(result['osVersion']))
    print(result['account'], type(result['account']))
    print(result['account']['username'])
    print(result['deviceName'])
    print(result['appPackage'])
    print(result['bool1'], type(result['bool1']))
    print(result['list'])
    print(result['tuple'])


# 读取多组数据
def test2():
    with open('../文件/yaml_test.yaml', 'r', encoding='utf-8') as f:
        result = yaml.load_all(f.read(), Loader=yaml.FullLoader)
        print(result, type(result))
        # 使用for循环遍历每一组数据
        for i in result:
            print(i)


# 写入单组数据
def test3():
    apiData = {
        "page": 1,
        "msg": "地址",
        "data": [{
            "id": 1,
            "name": "学校"
        }, {
            "id": 2,
            "name": "公寓"
        }, {
            "id": 3,
            "name": "流动人口社区"
        }],
    }

    with open('../文件/writeYamlData.yml', 'w', encoding='utf-8') as f:
        yaml.dump(data=apiData, stream=f, allow_unicode=True)


# 写入多组数据
def test4():
    import yaml

    apiData1 = {
        "page": 1,
        "msg": "地址",
        "data": [{
            "id": 1,
            "name": "学校"
        }, {
            "id": 2,
            "name": "公寓"
        }, {
            "id": 3,
            "name": "流动人口社区"
        }],
    }

    apiData2 = {
        "page": 2,
        "msg": "地址",
        "data": [{
            "id": 1,
            "name": "酒店"
        }, {
            "id": 2,
            "name": "医院"
        }, {
            "id": 3,
            "name": "养老院"
        }],
    }

    with open('../文件/writeYamlData.yml', 'w', encoding='utf-8') as f:
        yaml.dump_all(documents=[apiData1, apiData2], stream=f, allow_unicode=True)


if __name__ == '__main__':
    test4()