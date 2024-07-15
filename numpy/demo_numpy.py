"""
    numpy-python编程
"""

import numpy as np


# 简单创建一个数组
def test_np01():
    array = np.array([[1, 2, 3],
                      [2, 3, 4]])  # 定义一个矩阵
    print(array)
    print('维数', array.ndim)
    print('形状', array.shape)
    print('size', array.size)


# numpy创建各种数据
def test_np02():
    a = np.array([2, 3, 4], dtype=np.float64)  # 创建一个助阵，并定义数据类型
    print(a.dtype)
    print(a)

    b = np.array([[2, 3, 4],
                  [4, 5, 6]], dtype=np.float64)
    print(b.dtype)
    print(b)

    c = np.zeros((3, 4), dtype=np.int16)
    print(c)

    d = np.ones((3, 4), dtype=np.int16)
    print(d)

    e = np.empty((3, 4), dtype=np.float64)  # 生成一个接近与0的矩阵
    print(e)

    g = np.arange(10, 20, 2)  # 生成一个数列
    print(g)

    f = np.arange(12).reshape((3, 4))  # 将一个数列，变成一个矩阵
    print(f)

    g = np.linspace(0, 10, 6).reshape(2, 3)  # 分段函数
    print(g)


# numpy 基础运算（1）
def test_np03():
    a = np.array([10, 12, 13, 14])
    b = np.arange(4)
    print(a, b)
    # 加减运算，
    c = a - b
    print(c)
    # 乘除运算
    d = a * b
    print(d)
    # 幂次运算
    e = b ** 2
    print(e)
    # 三角函数运算 sin cos tan
    f = np.sin(a)
    print(f)
    # 判断矩阵中值与某个值的大小关系
    g = b < 3
    print(g)

    # 矩阵运算
    aa = np.array([[1, 2],
                   [3, 4]])
    bb = np.arange(4).reshape((2, 2))
    cc = aa * bb  # 普通乘法
    cc_dot = np.dot(aa, bb)  # 矩阵乘法
    cc_dot_2 = a.dot(b)  # 矩阵乘法第二种形式
    print(cc)
    print(cc_dot)

    r = np.random.random((2, 4))  # 传入矩阵的形状，随机生成0-1之间的数
    print(r)
    print(np.sum(r, axis=0))  # 矩阵就和 0列1行
    print(np.min(r, axis=1))  # 最小值
    print(np.max(r, axis=0))  # 最大值


# numpy 基础运算（2）
def test_np04():
    A = np.arange(2, 14).reshape((3, 4))
    print(A)
    print(np.argmin(A))  # 矩阵中最小值的位置
    print(np.argmax(A))  # 矩阵中最大值的位置
    print(np.average(A))  # 平均值
    print(np.average(A, axis=0))  # 列平均值
    print(np.average(A, axis=1))  # 行平均值
    print(np.median(A))  # 中位数
    print(np.cumsum(A))  # 逐个累加
    print(np.diff(A))  # 逐差
    print(np.nonzero(A))  # 找出非零的数，使用两个列表输出非零元素的位置
    print(np.sort(A))  # 将矩阵每一行排序
    print(np.transpose(A))  # 矩阵转置
    print(np.clip(A, 5, 9))  # 所有大于9的数变成9，小于5的数变成5


# numpy 矩阵索引
def test_np05():
    B = np.arange(2, 14).reshape((3, 4))
    print(B)
    print(B[2])  # 第二行的所有数
    print(B[:, 2])  # 第二列的所有数
    print(B[2, 3])  # 二行三列的一个数
    print(B[2, 1:3])  # 第二行第一个到第三个数
    # 迭代B的行
    for row in B:
        print(row)
    # 迭代B的列
    for column in B.T:
        print(column)
    # 迭代B的每一个元素
    print(B.flatten())  # 将矩阵变成一个数组
    for item in B.flat:
        print(item)


# numpy 矩阵合并
def test_np06():
    A = np.array([1, 1, 1])
    B = np.array([2, 2, 2])

    print(np.vstack((A, B)))  # 行合并矩阵
    print(np.hstack((A, B)))  # 列合并矩阵
    print(A[:, np.newaxis])  # 单行矩阵不能使用转置变成列矩阵，要这样变换。
    print(np.hstack((A[:, np.newaxis], B[:, np.newaxis])))

    print(np.concatenate((A[:, np.newaxis], B[:, np.newaxis],
                          B[:, np.newaxis], A[:, np.newaxis]), axis=1))  # 任意数量的矩阵合并，并且可以指定合并的方向


# numpy 矩阵分割
def test_np07():
    A = np.arange(12).reshape(3, 4)
    # 只能进行等量分割
    print(np.split(A, 2, axis=1))  # 纵向分割成两块
    print(np.split(A, 3, axis=0))  # 横向分割成三块
    print(np.vsplit(A, 3))  # 横向分割成三块
    print(np.hsplit(A, 2))  # 纵向分割成两块
    # 进行不等量分割
    print(np.array_split(A, 3, axis=1))


# numpy 矩阵拷贝
def test_np08():
    a = np.arange(4)
    # 以下的赋值方式并不会将拷贝一份给b，c，d,本质上都是a
    b = a
    c = a
    d = b
    # 深拷贝
    b = a.copy()  # b在一个全新的地址，和a无关


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    test_np07()
