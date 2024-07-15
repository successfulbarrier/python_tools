#!/usr/bin/env python
# -*- coding:utf-8 -*-
# project:Dive_into_deep_learning_2.0
# author:机灵巢穴_WitNest
# datetime:2023/9/2 21:10
# software: PyCharm

from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision
from torch.utils.data import DataLoader
import torch
from torch import nn
from d2l import torch as d2l
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


"""
    分类数据读取类
"""


class C_MyData(Dataset):

    def __init__(self, root_dir, label_dir, label_num, img_size, data_enhance="None"):
        """
        :param root_dir: 数据集根目录  ../train
        :param label_dir: 类别目录名称
        :param img_size: 图片尺寸 ，例如：（255，255）
        """
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.label_num = label_num
        self.path = os.path.join(root_dir, label_dir)
        self.img_path = os.listdir(self.path)
        if data_enhance == "None":
            self.trans_img = torchvision.transforms.Compose([torchvision.transforms.Resize(img_size),
                                                             torchvision.transforms.ToTensor()])
        elif data_enhance == "HorizontalFlip":
            self.trans_img = torchvision.transforms.Compose([torchvision.transforms.Resize(img_size),
                                                             torchvision.transforms.RandomHorizontalFlip(),
                                                             torchvision.transforms.ToTensor()])
        elif data_enhance == "ColorJitter":
            self.trans_img = torchvision.transforms.Compose([torchvision.transforms.Resize(img_size),
                                                             torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                                             torchvision.transforms.ToTensor()])
        elif data_enhance == "ResizedCrop":
            self.trans_img = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(img_size, scale=(0.1, 1), ratio=(0.5, 2)),
                                                             torchvision.transforms.ToTensor()])
        elif data_enhance == "all":
            self.trans_img = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                             torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                                             torchvision.transforms.RandomResizedCrop(img_size, scale=(0.1, 1), ratio=(0.5, 2)),
                                                             torchvision.transforms.ToTensor()])
        else:
            print("------------------>数据增强选择错误，将不做任何增强处理<-----------------")
            self.trans_img = torchvision.transforms.Compose([torchvision.transforms.Resize(img_size),
                                                             torchvision.transforms.ToTensor()])

    # 类对象迭代时调用的函数
    def __getitem__(self, item):
        img_name = self.img_path[item]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        img = self.trans_img(img)
        label = self.label_num
        return img, label

    def __len__(self):
        return len(self.img_path)


"""
    分类数据加载函数
"""


def C_data_load(root_dir, img_size, batch_size=32, data_enhance=[0]):
    """
    :param data_enhance:例如[1,2,3,4,4,4],选择范围0-4，0不增强，1左右翻转，2颜色调节，3随机裁剪，4全部增强
    :param batch_size: 批大小
    :param root_dir: 文件路径
    :param img_size: 想要统一转化的图片尺寸
    :return:训练集：train_all_dataset, 测试集：test_all_dataset, 类别列表：train_list
    """
    # 获取训练集和测试集的类别列表和路径
    train_path = os.path.join(root_dir, "train")
    test_path = os.path.join(root_dir, "test")
    train_list = os.listdir(train_path)
    test_list = os.listdir(test_path)

    # 遍历列表并筛选出文件夹名称
    train_list = [f for f in train_list if os.path.isdir(os.path.join(train_path, f))]
    test_list = [f for f in test_list if os.path.isdir(os.path.join(test_path, f))]

    # 检查训练集和测试集
    assert train_list == test_list, "测试集和训练集类别不同"

    # 实例化训练集各个类别数据集
    train_data_list = list()
    label_num = 0
    for folder_name in train_list:
        dataset = C_MyData(train_path, folder_name, label_num, img_size)
        train_data_list.append(dataset)
        label_num += 1

    # 实例化测试集各个类别数据集
    test_data_list = list()
    label_num = 0
    for folder_name in test_list:
        dataset = C_MyData(test_path, folder_name, label_num, img_size)
        test_data_list.append(dataset)
        label_num += 1

    # 数据增强部分
    for i in data_enhance:
        if i == 0:
            ...
        elif i == 1:
            label_num = 0
            for folder_name in train_list:
                dataset = C_MyData(train_path, folder_name, label_num, img_size, data_enhance="HorizontalFlip")
                train_data_list.append(dataset)
                label_num += 1
        elif i == 2:
            label_num = 0
            for folder_name in train_list:
                dataset = C_MyData(train_path, folder_name, label_num, img_size, data_enhance="ColorJitter")
                train_data_list.append(dataset)
                label_num += 1
        elif i == 3:
            label_num = 0
            for folder_name in train_list:
                dataset = C_MyData(train_path, folder_name, label_num, img_size, data_enhance="ResizedCrop")
                train_data_list.append(dataset)
                label_num += 1
        elif i == 4:
            label_num = 0
            for folder_name in train_list:
                dataset = C_MyData(train_path, folder_name, label_num, img_size, data_enhance="all")
                train_data_list.append(dataset)
                label_num += 1
        else:
            print("------------------>数据增强选择矩阵超范围<-----------------")
            return -1

    # 计算列表长度
    train_data_len = len(train_data_list)
    test_data_len = len(test_data_list)
    assert train_data_len > 1, "类别个数要大于等于2"

    # 累加训练集
    train_all_dataset = train_data_list[0] + train_data_list[1]
    for i in range(train_data_len-2):
        train_all_dataset = train_all_dataset + train_data_list[i+2]
    a = len(train_all_dataset)
    # 累加测试集
    test_all_dataset = test_data_list[0] + test_data_list[1]
    for i in range(test_data_len-2):
        test_all_dataset = test_all_dataset + test_data_list[i+2]

    train_iter = DataLoader(train_all_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_iter = DataLoader(test_all_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_iter, test_iter, train_list


"""
    分类网络训练函数
"""


def C_train_my(net, train_iter, test_iter, num_epochs, lr, device, optimizer="Adam"):
    """
    修改后的训练函数使用 tensorborad 显示数据
    :param net: 网络
    :param train_iter: 训练集
    :param test_iter: 测试集
    :param num_epochs: 训练轮数
    :param lr: 学习率
    :param device: 设备
    :param optimizer: 优化器，四种可选：SGD、Momentum、RMSprop、Adam
    :return:
    """

    # 初始化权重函数，全连接层和卷积层可用
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)     # 将参数初始化函数应用到整个网络上
    print('training on', device)
    net.to(device)  # 网络移动到GPU

    # 优化器选择
    if optimizer == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 定义优化器
        print("----------------->使用SGD优化器<---------------")
    elif optimizer == "Momentum":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.8)    # 定义优化器
        print("----------------->使用Momentum优化器<---------------")
    elif optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, alpha=0.9)
        print("----------------->使用RMSprop优化器<---------------")
    elif optimizer == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.99))
        print("----------------->使用Adam优化器<---------------")
    else:
        print("-------------请输入正确的优化器选项--------------")
        return -1

    loss = nn.CrossEntropyLoss()    # 定义损失函数

    timer, num_batches = d2l.Timer(), len(train_iter)     # 计算时间和训练集迭代次数
    writer = SummaryWriter()    # 定义tensorboard对象

    # 开始训练
    for epoch in tqdm(range(num_epochs)):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)     # 定义数据存储对象，传入的参数为存储数据的个数
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()   # 梯度清零
            X, y = X.to(device), y.to(device)   # 训练数据转移到GPU
            y_hat = net(X)  # 前向传播
            l = loss(y_hat, y)  # 损失计算
            l.backward()    # 计算梯度
            optimizer.step()    # 更新梯度
            timer.stop()

            # 计算各个参数
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])  # 分别存储这一个batch的 损失、准确的个数，总个数

            train_l = metric[0] / metric[2]     # 计算平均损失
            train_acc = metric[1] / metric[2]   # 计算准确率

            # 进度条显示1
            # print("\r", end="")
            # print("epoch {}: {}%: ".format(epoch, i*100//num_batches), "▋" * (i // 2), end="")
            # sys.stdout.flush()

            # tensorboard显示数据
            writer.add_scalar('loss', train_l, i + (num_batches * epoch))
            writer.add_scalar('train_acc', train_acc, i + (num_batches * epoch))

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)    # 测试集的准确率计算
        writer.add_scalar('test_acc', test_acc, epoch)

    writer.close()
    # 打印训练结果
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec（每秒训练的张数） '
          f'on {str(device)}')  # 每秒训练的张数


"""
    网络输出形状打印
"""


def Net_shape_print(net, input_shape):
    """
    :param net: 网络传入
    :param input_shape: 网络输入张量形状，(1, 1, 28, 28)
    :return:
    """
    X = torch.rand(size=input_shape, dtype=torch.float32)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape: \t', X.shape)


"""
    使用 tensorborad 显示网络结构
"""


def TB_print_net(net, input_size):
    writer = SummaryWriter()
    x = torch.rand(size=input_size, dtype=torch.float32)
    writer.add_graph(net, x)
    writer.close()


"""
    测试主函数
"""


if __name__ == '__main__':
    train_dataset, test_dataset, class_list = C_data_load("D:\\my_file\\data_set\\classification\\flowers", [300, 300])
    print(len(train_dataset))
    print(len(test_dataset))
    print(class_list)
    # print(train_dataset[0][0].shape)
    # train_dataloder = DataLoader(train_dataset, batch_size=64)
    # test_dataloder = DataLoader(test_dataset, batch_size=64)