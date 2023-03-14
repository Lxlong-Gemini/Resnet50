import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from ResNet50_model2 import *

# 训练和测试数据集的准备
from torch.utils.data import DataLoader

train_data = torchvision.datasets.VOCDetection("D:\\AppGallery\\python文件\\ResNet50(lxl)", "2012",
                                               "train", False, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.VOCDetection("D:\\AppGallery\\python文件\\ResNet50(lxl)", "2012",
                                               "test", False, transform=torchvision.transforms.ToTensor())
# 训练和测试数据集的长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}", format(train_data_size))
print("测试数据集的长度为：{}", format(test_data_size))

# 利用DataLoader来加载数据
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
net = ResNet()

# 损失函数
lossfunction = nn.MultiLabelSoftMarginLoss

# 优化器
learning_rate = 1e-3
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# 训练和测试的次数
total_train_step = 0
total_test_step = 0
# 训练的轮数
epoch = 10

# 将训练和测试的损失写入tensorboard
# writer = SummaryWriter("D:\\AppGallery\\python文件\\n_network\\logs")

for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i+1))

    # 训练步骤开始
    net.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = net(imgs)
        loss = lossfunction(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数:{}, Loss:{}".format(total_train_step, loss.item()))
            # writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    net.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = net(imgs)
            loss = lossfunction(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体数据集上的Loss:{}".format(total_test_loss))
    print("整体数据集上的准确率:{}".format(total_accuracy/test_data_size))
    # writer.add_scalar("test_loss", total_test_loss, total_test_step)
    # writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_train_step)
    total_test_step = total_test_step + 1

    torch.save(net, "lxl_{}.pth".format(i))

# writer.close()
