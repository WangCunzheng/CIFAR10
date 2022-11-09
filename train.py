import time
import torch
import torchvision
from torch import nn
from model import net
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# 定义训练的设备,单显卡cuda与cuda:0一样，多显卡要定义
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 获取长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为: {} ".format(train_data_size))
print("测试数据集的长度为: {} ".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
sz = net.ShouZheng()
# 模型和损失函数的to(device)不需要另外赋值
sz.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 优化器
learning_rate = 1e-3
optimizer = torch.optim.Adam(sz.parameters(), lr=learning_rate, weight_decay=0.01)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 100
# 记录每次train loss
train_loss = []

# 添加tensorboardX
writer = SummaryWriter("./logs_train")
start_time = time.time()

for i in range(epoch):
    print("--------第 {} 轮训练开始--------".format(i + 1))

    # 训练步骤开始
    sz.train()
    total_train_loss = 0
    for data in train_dataloader:
        imgs, targets = data
        # 图片和标注的to(device)需要另外赋值
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = sz(imgs)
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss = total_train_loss + loss.item()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("时间：{}".format(end_time - start_time))
            print("训练次数： {}， Loss： {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    train_loss.append(total_train_loss)

    # 测试步骤开始
    sz.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = sz(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = accuracy + total_accuracy

    print("整体测试集上的 Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1
    torch.save(sz, "./src/sz_{}.pth".format(i))
    # torch.save(sz.state_dict(), "sz_{}.pth".format(i))
    print("模型已保存")

writer.close()
a = min(train_loss)  # 最小值
b = train_loss.index(min(train_loss))  # 最小值的位置
print("训练最小loss为： {}最文件为： sz_{}.pth".format(a, b))
# 测试步骤开始
sz_net = torch.load("./src/sz_" + str(b) + ".pth")
sz_net.eval()
total_test_loss = 0
total_accuracy = 0
with torch.no_grad():
    for data in test_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = sz_net(imgs)
        loss = loss_fn(outputs, targets)
        total_test_loss = total_test_loss + loss.item()
        accuracy = (outputs.argmax(1) == targets).sum()
        total_accuracy = accuracy + total_accuracy

print("整体测试集上的 Loss: {}".format(total_test_loss))
print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))

writer.add_scalar("test_loss", total_test_loss, total_test_step)
writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
total_test_step = total_test_step + 1

# 目前模型75最好69%
