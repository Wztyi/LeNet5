import copy
import pandas as pd
import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet
import torch.nn as nn
import time


def train_val_process():
    datasets = FashionMNIST(
        root="./data", train=True, transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]), download=True
    )
    train_data, val_data = Data.random_split(datasets, [round(0.8 * len(datasets)), round(0.2 * len(datasets))])
    train_dataloader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=0)
    val_dataloader = Data.DataLoader(dataset=val_data, batch_size=128, shuffle=True, num_workers=0)
    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    # 设定训练所用到设备，有GPU的用GPU，没有则用CPU来进行训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用Adam优化器，， 学习率设置为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 定义交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 将模型放到设备当中
    model = model.to(device)
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失函数列表
    train_loss_all = []
    # 验证集损失函数列表
    val_loss_all = []
    # 训练集精度列表
    train_acc_all = []
    # 验证集精度列表
    val_acc_all = []
    # 当前时间
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # 初始化参数
        # 初始化损失函数
        train_loss = 0.0
        # 训练集准确度
        train_corrects = 0
        # 验证集损失函数
        val_loss = 0.0
        # 验证集准确度
        val_corrects = 0
        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将特征放入到训练设备中
            b_x = b_x.to(device)
            # 将标签放入到训练设置中
            b_y = b_y.to(device)
            # 设置模型为训练模式
            model.train()
            # 前向传播过程,输入一个btch,输出一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的下标
            pre_lab = torch.argmax(output, dim=1)

            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)

            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向传播计算
            loss.backward()
            # 根据网络反向传播的梯度信息来更新网络的参数,以起到降低loss函数计算值的作用
            optimizer.step()
            # 对损失函数进行累加
            train_loss += loss.item() * b_x.size(0)
            # 如果预测结果正确,则准确度train_corrects + 1
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用户训练的样本数量
            train_num += b_x.size(0)
        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将特征放入到训练设备中
            b_x = b_x.to(device)
            # 将标签放入到训练设置中
            b_y = b_y.to(device)
            # 设置模型为评估模式
            model.eval()

            # 前向传播过程,输入一个batch,输出为一个batch中对应的预测
            output = model(b_x)

            # 查找每一行中最大值对应的下标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)

            # 对损失函数进行累加
            val_loss += loss.item() * b_x.size(0)
            # 如果预测结果正确,则准确度train_corrects + 1
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用户验证的样本数量
            val_num += b_x.size(0)
        # 计算并保存每一次迭代的loss值和准确率
        train_loss_all.append(train_loss / train_num)
        # 计算并保存训练集的准确率
        train_acc_all.append(train_corrects.double().item() / train_num)

        # 计算并保存验证集的loss值
        val_loss_all.append(val_loss / val_num)
        # 计算并保存验证集的准确率
        val_acc_all.append(val_corrects.double().item() / val_num)

        print("{} Train Loss: {:.4f} Train Acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} Train Loss: {:.4f} Train Acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        # print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))

        # 寻找最高的准确度权重参数
        if val_acc_all[-1] > best_acc:
            # 保存当前的最高准确度
            best_acc = val_acc_all[-1]
            # 保存当前的最高准确度
            best_model_wts = copy.deepcopy(model.state_dict())
        # 训练耗时
        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

    # 选择最优参数
    # 加载最高准确率下的参数模型
    # model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, "./runs/best_model.pth")
    # torch.save(model.state_dict(best_model_wts), "./runs/best_model.pth")
    train_process = pd.DataFrame(
        data={
            "epoch": range(num_epochs),
            "train_loss_all": train_loss_all,
            "val_loss_all": val_loss_all,
            "train_acc_all": train_acc_all,
            "val_acc_all": val_acc_all,
        }
    )

    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, "ro-", label="train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, "bs-", label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.xlabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, "ro-", label="train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, "bs-", label="val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.xlabel("acc")
    plt.legend()
    plt.savefig("result.png")
    plt.show()


if __name__ == "__main__":
    # 实例化模型
    LeNet = LeNet()
    print(123)
    train_dataloader, val_dataloader = train_val_process()
    train_process = train_model_process(LeNet, train_dataloader, val_dataloader, 40)
    matplot_acc_loss(train_process)
