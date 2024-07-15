"""
    多线程python编程
"""


import threading
import time
from queue import Queue


# 第一个例子
def threading_01():
    print(threading.active_count())  # 当前运行的线程数
    print(threading.enumerate())  # 当前正在运行的线程名称
    print(threading.current_thread())  # 这条语句运行在那个线程


# 第二个例子
def thread_job():
    print('this is added thread,number is %s' % threading.current_thread())
    for i in range(10):
        time.sleep(0.5)
        print('.....')


def threading_02():
    added_thread = threading.Thread(target=thread_job, name='Thread')  # 定义一个线程
    added_thread.start()  # 开始运行线程
    added_thread.join()  # 等待该线程运行结束
    print('线程运行完毕')


# 第三个例子:多线程提升计算速度，必须使用队列存储数据
def job(l, q):
    for i in range(len(l)):
        l[i] = l[i] ** 2
    q.put(l)


def multithreading():
    q = Queue()
    threads = []
    data = [[1, 2, 3], [3, 4, 5], [4, 4, 4], [5, 5, 5]]
    for i in range(4):
        t = threading.Thread(target=job, args=(data[i], q))
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()
    results = []
    for _ in range(4):
        results.append(q.get())
    print(results)


# 第四个例子，lock锁
# 但不同线程操作全局变量时使用
A = 0
lock = threading.Lock()


def job1():
    global A,lock
    lock.acquire()
    for i in range(10):
        A += 1
        print('job1', A)
    lock.release()


def job2():
    global A,lock
    lock.acquire()
    for i in range(10):
        A += 10
        print('job2', A)
    lock.release()


def threading_04():
    added_thread1 = threading.Thread(target=job1, name='job1')  # 定义一个线程
    added_thread2 = threading.Thread(target=job2, name='job2')  # 定义一个线程
    added_thread1.start()  # 开始运行线程
    added_thread2.start()  # 开始运行线程
    added_thread1.join()  # 等待该线程运行结束
    added_thread2.join()  # 等待该线程运行结束
    print('线程执行完毕')


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    threading_04()
