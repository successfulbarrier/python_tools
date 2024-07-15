"""
    多进程python编程
    多进程可以加快数据处理的速度，也可以并行计算，多核运算真正的并行运算。
"""
import multiprocessing as mp

# 第一个例子
import time


def job(a, d):
    print('aaaaaa')


# 与多线程的创建方式一致
def mp_01():
    p1 = mp.Process(target=job, args=(1, 2))  # 定义进程
    p1.start()  # 开始进程
    p1.join()  # 等待进程结束


# 第二个例子
def job2(q):
    res = 0
    for i in range(10000):
        res += i + i ** 2 + i ** 3
    q.put(res)


# 使用Queue()方法创建队列，用于存放不同进程的数据
def mp_02():
    q = mp.Queue()  # 队列存放多进程中运算完的数据（必须使用这种方式）
    p1 = mp.Process(target=job2, args=(q,))  # 定义进程
    p2 = mp.Process(target=job2, args=(q,))  # 定义进程
    p1.start()  # 开始进程
    p2.start()  # 开始进程
    p1.join()  # 等待进程结束
    p2.join()  # 等待进程结束
    print(q.get())
    print(q.get())


# 第三个例子 （使用进程池）
def job3(x):
    return x * x


def multicore_1():
    pool = mp.Pool(processes=3)

    # 单个参数时可以使用比较简单，直接一下子扔进去，多进程执行
    res = pool.map(job3, range(10))
    print(res)

    # 参数较为复杂时使用，只能一次执行一个参数
    res = pool.apply_async(job3, (2,))
    print(res.get())

    # 通过迭代实现循环开启多个进程
    multi_res = [pool.apply_async(job3, (i,)) for i in range(10)]  # 迭代执行某项操作
    print([res.get() for res in multi_res])


# 第四个例子 （使用进程池）
# 多进程之间变量是不可以共用的，必须使用共享内存来进行访问。
def mp_03():
    value = mp.Value('d', 1)  # 定义一个数的共享内存
    array = mp.Array('i', [1, 2, 3])  # 定义一个数组的共享内存，只能是一维数组。
    print(value.value)
    i = 0
    a = [print(array[i]) for i in range(3)]
    print(a)


# 第四个例子 （使用锁访问共享内存）
def job4(v, num, l):
    l.acquire()
    for _ in range(10):
        time.sleep(0.2)
        v.value += num
        print(v.value)
    l.release()


def mp_04():
    v = mp.Value('i', 0)  # 定义一个数的共享内存
    l = mp.Lock()  # 定义进程锁
    p1 = mp.Process(target=job4, args=(v, 1, l))  # 定义进程
    p2 = mp.Process(target=job4, args=(v, 3, l))  # 定义进程
    p1.start()  # 开始进程
    p2.start()  # 开始进程
    p1.join()  # 等待进程结束
    p2.join()  # 等待进程结束


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    mp_04()
