"""
    pandas-python编程
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 简单创建一个简单的表格
def test_np01():
    s = pd.Series([1, 2, 3, np.nan, 44, 1])  # 产生一个自定义序列
    print(s)
    dates = pd.date_range('20230101', periods=6)  # 根据你给的数据类型产生一个序列
    print(dates)
    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['a', 'b', 'c', 'd'])  # 定义矩阵的行和列的名称
    print(df)
    df1 = pd.DataFrame(np.arange(12).reshape((3, 4)))
    print(df1)
    df2 = pd.DataFrame({'A': 1,
                        "B": pd.Timestamp('20230101'),
                        'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                        'D': np.array([3] * 4, dtype='int32'),
                        'E': pd.Categorical(['test', 'train', 'test', 'train']),
                        'F': 'foo'})  # 使用字典的形式定义
    print(df2)
    print(df2.dtypes)  # 矩阵每一列的类型
    print(df2.index)  # 矩阵行的序号
    print(df2.columns)  # 矩阵的列号
    print(df2.values)  # 矩阵的值
    print(df2.describe())  # 只能显示数字数据的运算结果
    print(df2.T)  # 转置表格
    print(df2.sort_index(axis=1, ascending=False))  # 将列进行倒着排序
    print(df2.sort_values(by='E'))  # 按列的内容进行排序


# pandas 数据筛选
def test_np02():
    dates = pd.date_range('20230101', periods=6)
    df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', "C", "D"])
    print(df)
    print(df.A)
    print(df[0:3], df['20230102':'20230104'])
    print("--------------")
    print(df.loc['20230102'])  # 筛选指定行
    print("--------------")
    print(df.loc[:, ['A', 'B']])  # 筛选指定列
    print("--------------")
    print(df.iloc[3:5, 1:3])  # 筛选指定行和列
    # print(df.ix[:3, ['A', 'C']])
    print(df)
    print(df[df.A < 8])  # 按某列数据的大小比较筛选


# pandas 设置值
def test_np03():
    dates = pd.date_range('20230101', periods=6)
    df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', "C", "D"])
    # 修改某位置的值
    df.iloc[2, 2] = 111  # 按位置定位
    print(df)
    df.loc['20230101', 'B'] = 222  # 按标签定位
    print(df)
    # 通过比较修改值
    df[df.A > 4] = 0  # 把A的那一列大于4的哪些行都变成0
    print(df)
    df.A[df.A > 4] = 0  # 只把A哪一行大于4的值变成0
    print(df)
    # 添加新的行或者列
    df['F'] = np.nan
    df['E'] = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20230101', periods=6))
    print(df)


# pandas 处理丢失的数据
def test_np04():
    dates = pd.date_range('20230101', periods=6)
    df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', "C", "D"])
    df.iloc[2, 2] = np.nan
    df.iloc[3, 1] = np.nan
    print(df)
    # print(df.dropna(axis=0, how='any'))  # how='all'全是nan时丢掉， 1丢列
    # print(df.fillna(value=0))  # 把所有缺失的值都赋值为0
    print(df.isnull())  # 查询表格中哪一个地方的数据缺失
    print(np.any(df.isnull()))
    print(df.fillna(value=0))  # 把所有缺失的值都赋值为0
    print(df)


# pandas 数据导入导出
def test_np05():
    data = pd.read_csv('../文件/student.csv')
    print(data)
    data.to_pickle('.\文件\student.pickle')
    # data.to_excel('.\文件\student.xlsx')
    data.to_csv('.\文件\student2.csv')


# pandas 合并concat
def test_np06():
    df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'])
    df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['a', 'b', 'c', 'd'])
    df3 = pd.DataFrame(np.ones((3, 4)) * 2, columns=['a', 'b', 'c', 'd'])
    print(df1)
    print(df2)
    print(df3)
    res = pd.concat([df1, df2, df3], axis=0, ignore_index=True)  # 行合并, ignore_index=True 重新排序
    print(res)
    df5 = pd.DataFrame(np.ones((3, 4)) * 3, columns=['a', 'b', 'c', 'd'], index=[1, 2, 3])
    df6 = pd.DataFrame(np.ones((3, 4)) * 4, columns=['b', 'c', 'd', 'e'], index=[2, 3, 4])
    print(df5)
    print(df6)
    ress = pd.concat([df5, df6], axis=0, join='outer', ignore_index=True)
    print(ress)
    ress = pd.concat([df5, df6], axis=0, join='inner', ignore_index=True)
    print(ress)
    ress = pd.concat([df5, df6], axis=1).iloc[[0, 1, 2]]  # 选择你需要的行
    print(ress)


# pandas 合并merge
def test_np07():
    left = pd.DataFrame({'Key': ['K0', 'K1', 'K2', 'K3'],
                         'A': ['A0', 'A1', 'A2', 'A3'],
                         'B': ['B0', 'B1', 'B2', 'B3']})
    right = pd.DataFrame({'Key': ['K0', 'K1', 'K2', 'K3'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']})
    print(left)
    print(right)
    res = pd.merge(left, right, on='Key')
    print(res)

    left = pd.DataFrame({'Key1': ['K0', 'K1', 'K0', 'K3'],
                         'Key2': ['K0', 'K1', 'K1', 'K3'],
                         'A': ['A0', 'A1', 'A2', 'A3'],
                         'B': ['B0', 'B1', 'B2', 'B3']})
    right = pd.DataFrame({'Key1': ['K0', 'K1', 'K2', 'K3'],
                          'Key2': ['K0', 'K0', 'K0', 'K3'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']})
    res = pd.merge(left, right, on=['Key1', 'Key2'], how='inner')  # 只会把相同的部分进行合并(交集)
    print(res)
    res = pd.merge(left, right, on=['Key1', 'Key2'], how='outer')  # 只会把相同的部分进行合并（并集）
    print(res)
    res = pd.merge(left, right, on=['Key1', 'Key2'], how='left')  # 只会把相同的部分进行合并(基于left)
    print(res)
    res = pd.merge(left, right, on=['Key1', 'Key2'], how='right')  # 只会把相同的部分进行合并（基于right）
    print(res)

    # 合并两个表格并并产生一个标签页
    df1 = pd.DataFrame({'col1': [0, 1], 'col_left': ['a', 'b']})
    df2 = pd.DataFrame({'col1': [1, 2, 2], 'col_right': [2, 2, 2]})
    print(df1)
    print(df2)
    res = pd.merge(df1, df2, on='col1', how='outer', indicator=True)
    print(res)
    res = pd.merge(df1, df2, on='col1', how='outer', indicator='indicator_column')
    print(res)

    #
    left = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                         'B': ['B0', 'B1', 'B2', 'B3'],
                         'index': ['K0', 'K1', 'K2', 'K3']})
    right = pd.DataFrame({'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3'],
                          'index': ['K0', 'K1', 'K2', 'K3']})
    print(left)
    print(right)
    res = pd.merge(left, right, left_index=True, right_index=True, how='outer')
    print(res)

    # 将两个表格中相同列表的标签加不同前缀加以区分
    boys = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'age': [1, 2, 3]})
    girls = pd.DataFrame({'k': ['K0', 'K1', 'K3'], 'age': [1, 2, 3]})
    print(boys)
    print(girls)
    res = pd.merge(boys, girls, on='k', suffixes=['_boy', 'girl'], how='inner')
    print(res)

    # 使用 matplotlib.pyplot 库绘图
    data = pd.Series(np.random.randn(1000), index=np.arange(1000))
    data = data.cumsum()
    # 将数据绘图
    data.plot()
    plt.show()

    data = pd.DataFrame(np.random.randn(1000, 4),
                        index=np.arange(1000),
                        columns=list("ABCD"))
    data = data.cumsum()
    data.plot()
    plt.show()
    # print(data.head())

    # 图像类型
    # bar, hist box kde area scatter hexbin pie
    ax = data.plot.scatter(x='A', y='B', color='DarkBlue', label='Class 1')
    data.plot.scatter(x='A', y='C', color='DarkGreen', label='Class 2', ax=ax)
    plt.show()


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    test_np02()
