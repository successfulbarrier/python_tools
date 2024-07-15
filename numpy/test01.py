import numpy as np


# 0-24
def function01():
    # 版本信息
    print(np.__version__)
    # 配置信息
    # np.show_config()

    # 创建空向量
    Z = np.zeros(10)
    print(Z)
    # 获取向量所占内存大小
    Z = np.zeros((10, 10))
    print("%d bytes" % (Z.size * Z.itemsize))
    # 获取np.add的说明文档
    # np.info(np.add)

    # 创建一个值域范围在10~49的向量
    Z = np.arange(10, 50)
    print(Z)
    # 反转一个向量
    Z = np.arange(50)
    Z = Z[::-1]
    print(Z)

    # 将向量改变成矩阵，及其反过程
    Z = np.arange(9).reshape(3, 3)
    print(Z)
    # 寻找向量中非零元素的位置索引（矩阵不行）
    nz = np.nonzero(np.arange(9))
    print(nz)
    # 创建单位阵
    Z = np.eye(3)
    print(Z)

    # 创建一个3*3*3的随机数组
    Z = np.random.random((3, 3, 3))
    print(Z)
    # 寻找矩阵的最大最小值
    Z = np.random.random((10, 10))
    Zmin, Zmax = Z.min(), Z.max()
    print(Zmin, Zmax)
    # 寻找向量平均值
    Z = np.random.random(30)
    m = Z.mean()
    print(m)
    # 设置值1，2，3，4落在矩阵的对角线下方
    Z = np.diag(1 + np.arange(4), k=-1)
    print(Z)

    # 创建一个8*8的矩阵，并且设置成棋盘样式
    Z = np.zeros((8, 8), dtype=int)
    Z[1::2, ::2] = 1
    Z[::2, 1::2] = 1
    print(Z)
    # 三维数组索引
    Z = np.random.random((6, 7, 8))
    print(np.unravel_index(100, (6, 7, 8)))
    print(Z)
    # 创建一个8*8的矩阵，并且设置成棋盘样式
    Z = np.tile(np.array([[0, 1], [1, 0]]), (4, 4))
    print(Z)
    # 对随机矩阵做归一化处理
    Z = np.random.random((5, 5))
    Zmax, Zmin = Z.max(), Z.min()
    Z = (Z - Zmin) / (Zmax - Zmin)
    print(Z)
    # 自定义数据类型
    color = np.dtype([("r", np.ubyte, 1),
                      ("g", np.ubyte, 1),
                      ("b", np.ubyte, 1),
                      ("a", np.ubyte, 1)])
    print(color)

    # 矩阵乘法
    Z = np.dot(np.ones((5, 3)), np.ones((3, 2)))
    print(Z)


# 25-50
def function02():
    # 操作指定区间的元素
    Z = np.arange(11)
    Z[(3 < Z) & (Z <= 8)] *= -1
    print(Z)

    # 普通求和和numpy模块求和函数的区别
    print(sum(range(5), -1))
    print(np.sum(range(5), -1))
    # 将浮点数舍入到整数
    Z = np.random.uniform(-10, +10, 10)
    print(Z)
    print(np.copysign(np.ceil(np.abs(Z)), Z))
    # 找到两个数组中的共同元素
    Z1 = np.random.randint(0, 10, 10)
    Z2 = np.random.randint(0, 10, 10)
    print(np.intersect1d(Z1, Z2))

    # 获取日期信息
    yesterday = np.datetime64('today', 'D') - np.datetime64(1, 'D')
    today = np.datetime64('today', 'D')
    # tomorrow = np.datetime64('today', 'D') + np.datetime64(1, 'D')
    print("Yesterday is " + str(yesterday))
    print("today is " + str(today))
    # print("tomorrow is" + str(tomorrow))

    # 得到某一段日期
    Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
    print(Z)

    # 直接在位计算（A+B）*(-A/2)
    A = np.ones(3) * 1
    B = np.ones(3) * 2
    C = np.ones(3) * 3
    np.add(A, B, out=B)
    np.divide(A, 2, out=A)
    np.negative(A, out=A)
    np.multiply(A, B, out=A)
    print(A)

    # 五种方法提取一个随机数组的整数部分
    Z = np.random.uniform(0, 10, 10)
    print(Z - Z % 1)
    print(np.floor(Z))
    print(np.ceil(Z) - 1)
    print(Z.astype(int))
    print(np.trunc(Z))

    # 创建一个5*5的矩阵每行数的范围从0~4
    Z = np.zeros((5, 5))
    Z += np.arange(5)
    print(Z)

    # 创建一个长度为10的随机向量，其值域范围从0~1，但是不包括0和1
    Z = np.linspace(0, 1, 11, endpoint=False)[1:]
    print(Z)

    # 创建长度为10的随机向量并排序
    Z = np.random.random(10)
    Z.sort()
    print(Z)

    # 更快的方式对小数组求和
    Z = np.arange(10)
    sum1 = np.add.reduce(Z)
    print(sum1)

    # 检查两个数组是否相等
    A = np.random.randint(0, 2, 5)
    B = np.random.randint(0, 2, 5)
    equal = np.allclose(A, B)
    print(equal)

    # 创建一个只读数组
    Z = np.zeros(10)
    Z.flags.writeable = False

    # 将直角坐标系下的坐标转化为极坐标系下的坐标
    Z = np.random.random((10, 2))
    X, Y = Z[:, 0], Z[:, 1]
    R = np.sqrt(X ** 2 + Y ** 2)
    T = np.arctan2(Y, X)
    print(R)
    print(T)

    # 修改向量中的最大值
    Z = np.random.random(10)
    Z[Z.argmax()] = 1
    print(Z)

    # 创建一个结构化数组，并实现x和y坐标覆盖[0,1]*[0,1]区域
    Z = np.zeros((5, 5), [('x', float), ('y', float)])
    Z['x'], Z['y'] = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
    print(Z)

    # 给定两个数组X和Y，构造柯西矩阵。
    X = np.arange(8)
    Y = X + 0.5
    C = 1.0 / np.subtract.outer(X, Y)
    print(np.linalg.det(C))

    # 每个标量类型的最大值和最小值
    for dtype in [np.int8, np.int32, np.int64]:
        print(np.iinfo(dtype).min)
        print(np.iinfo(dtype).max)

    for dtype in [np.float32, np.float64]:
        print(np.finfo(dtype).min)
        print(np.finfo(dtype).max)
        print(np.finfo(dtype).eps)

    # 打印数组中的所有数值
    # np.set_printoptions(threshold=np.nan)
    Z = np.zeros((16, 16))
    print(Z)

    # 如何找到数组中最接近标量的值
    Z = np.arange(100)
    v = np.random.uniform(0, 100)
    index = (np.abs(Z - v)).argmin()
    print(v)
    print(Z[index])
    # np.info(np.random.uniform)


# 50-75
def function03():
    # 创建一个表示位置（x,y）和颜色（r,g,b）的结构化数组
    Z = np.zeros(10, [('position', [('x', float, 1), ('y', float, 1)]),
                      ('color', [('r', float, 1), ('g', float, 1), ('b', float, 1)])])
    print(Z)

    # 求每个点个所有点之间的距离
    Z = np.random.random((10, 2))
    X, Y = np.atleast_2d(Z[:, 0], Z[:, 1])
    D = np.sqrt((X - X.T) ** 2 + (Y - Y.T) ** 2)
    print(D)

    # 将32位整数转化为32位浮点数
    Z = np.arange(10, dtype=np.int32)
    Z = Z.astype(np.float32, copy=False)
    print(Z)

    # 读取文件

    #
    Z = np.arange(9).reshape(3, 3)
    for index, value in np.ndenumerate(Z):
        print(index, value)

    for index in np.ndindex(Z.shape):
        print(index, Z[index])

    # 生成一个通用的二维Gaussian-like数组
    X, Y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    D = np.sqrt(X * X + Y * Y)
    sigma, mu = 1.0, 0.0
    G = np.exp(-((D - mu) ** 2 / (2.0 * sigma ** 2)))
    print(G)

    # 在二维数组内部随机放置p个元素
    n = 10
    p = 3
    Z = np.zeros((n, n))
    np.put(Z, np.random.choice(range(n * n), p, replace=False), 1)
    print(Z)

    # 减去一个矩阵中的每一行的平均值
    X = np.random.rand(5, 10)
    Y = X - X.mean(axis=1, keepdims=True)
    print(Y)

    # 如何通过第n列对一个数组进行排序
    Z = np.random.randint(0, 10, (3, 3))
    print(Z)
    print(Z[Z[:, 1].argsort()])

    # 检查一个二维数组是否有空列
    Z = np.random.randint(0, 3, (3, 10))
    print((~Z.any(axis=0)).any())

    # 从数组中的给定值中找出最接近的值
    Z = np.random.uniform(0, 1, 10)
    z = 0.5
    m = Z.flat[np.abs(Z - z).argmin()]
    print(m)

    # 使用迭代器计算两个分别具有形状（1，3）和（3，1）的数组
    A = np.arange(3).reshape(3, 1)
    B = np.arange(3).reshape(1, 3)
    it = np.nditer([A, B, None])
    for x, y, z in it:
        z[...] = x + y
        print(it.operands[2])

    # 创建一个具有name属性的数组类
    class NamedArray(np.ndarray):
        def __new__(cls, array, name="no name"):
            obj = np.asarray(array).view(cls)
            obj.name = name
            return obj

        def __array_finalize__(self, obj):
            if obj is None: return
            self.info = getattr(obj, 'name', "no name")

    Z = NamedArray(np.arange(10), "range_10")
    print(Z)
    print(Z.name)

    # 如何对由第二个向量索引的每个元素加1
    Z = np.ones(10)
    I = np.random.randint(0, len(Z), 20)
    Z += np.bincount(I, minlength=len(Z))
    print(Z)

    # 根据索引列表（I）,如何将向量X的元素累加到数组（F）？
    X = [1, 2, 3, 4, 5, 6]
    I = [1, 3, 9, 3, 4, 1]
    F = np.bincount(I, X)
    print(F)

    # 考虑一个（dtype = ubyte）的（w,h,3）图像，计算其唯一颜色的数量。
    w, h = 16, 16
    I = np.random.randint(0, 2, (h, w, 3)).astype(np.ubyte)
    F = I[..., 0]*(256*256) + I[..., 1]*256 + I[..., 2]
    n = len(np.unique(F))
    print(n)

    # 四维数组，如何一次性计算出最后两个轴的和？
    A = np.random.randint(0, 10, (3, 4, 3, 4))
    sum = A.sum(axis=(-2, -1))
    print(sum)

    # 考虑一个一维向量D，如何使用相同大小的向量S来计算D子集的均值？
    D = np.random.uniform(0, 1, 100)
    S = np.random.randint(0, 10, 100)
    D_sums = np.bincount(S, weights=D)
    D_counts = np.bincount(S)
    D_means = D_sums/D_counts
    print(D_means)

    # 获取点积的对角线
    A = np.random.uniform(0, 1, (5, 5))
    B = np.random.uniform(0, 1, (5, 5))
    np.diag(np.dot(A, B))
    Z = np.sum(A*B.T, axis=1)
    print(Z)

    # 考虑一个向量[1,2,3,4,5],如何建立一个新的向量，在这个新向量中每个值之间有3个连续的零、
    Z = np.array([1,2,3,4,5])
    nz = 3
    Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
    Z0[::nz+1] = Z
    print(Z0)

    # 考虑一个维度（5，5，3）的数组，如何与一个（5，5）的数组相乘
    A = np.ones((5,5,3))
    B = 2*np.ones((5,5))
    print(A*B[:, :, None])

    # 如何对一个数组中的任意两行座交换
    A = np.arange(25).reshape(5, 5)
    A[[0, 1]] = A[[1, 0]]
    print(A)

    # 考虑一个可以描述10个三角形的triplets,找到可以分割全部三角形的line segment
    faces = np.random.randint(0, 100, (10, 3))
    F = np.roll(faces.repeat(2, axis=1), -1, axis=1)
    F = F.reshape(len(F)*3, 2)
    F = np.sort(F, axis=1)
    G = F.view(dtype=[('p0', F.dtype), ('p1', F.dtype)])
    G = np.unique(G)
    print(G)

    # 给定一个二进制的数组C，如何产生一个数组A满足np.bincount(A) == C
    C = np.bincount([1,1,2,3,4,4,6])
    A = np.repeat(np.arange(len(C)), C)
    print(A)

    #
    Z = np.random.randint(0, 2, 100)
    np.logical_not(Z, out=Z)
    Z = np.random.uniform(-1.0, 1.0, 100)
    np.negative(Z, out=Z)

    # 考虑两组点集P0和P1去描述一组（二维）和一个点p，如何计算点p到每一条线i的距离。
    




# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    function03()
