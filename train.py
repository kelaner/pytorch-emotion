from __future__ import print_function, division

import os

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from net import ThreeConvNet  # net.py文件中定义了ThreeConvNet网络结构

writer = SummaryWriter('logs')


# 训练主函数
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    for epoch in range(num_epochs):  # 迭代训练的轮数
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        for phase in ['train', 'val']:  # 对训练集和验证集分别进行操作
            if phase == 'train':
                model.train(True)  # 设置模型为训练模式
            else:
                model.train(False)  # 设置模型为评估模式

            running_loss = 0.0  # 累积损失值
            running_accs = 0.0  # 累积准确率
            number_batch = 0  # 记录处理的批次数

            # 遍历数据加载器中的数据
            for data in dataloaders[phase]:
                inputs, labels = data  # 获取输入数据和标签
                if use_gpu:
                    inputs = inputs.cuda()  # 如果使用GPU，则将数据转移到GPU上
                    labels = labels.cuda()

                optimizer.zero_grad()  # 梯度归零
                outputs = model(inputs)  # 前向传播得到输出
                _, preds = torch.max(outputs.data, 1)  # 得到预测结果
                loss = criterion(outputs, labels)  # 计算损失
                if phase == 'train':
                    loss.backward()  # 反向传播
                    optimizer.step()  # 更新权重
                    scheduler.step()  # 更新学习率

                running_loss += loss.data.item()  # 累加损失
                running_accs += torch.sum(preds == labels).item()  # 累加正确预测的数量
                number_batch += 1  # 批次数加一

            # 计算平均损失和准确率
            epoch_loss = running_loss / number_batch
            epoch_acc = running_accs / dataset_sizes[phase]

            # 使用tensorboard记录训练集和验证集的损失和准确率
            if phase == 'train':
                writer.add_scalar('data/trainloss', epoch_loss, epoch)
                writer.add_scalar('data/trainacc', epoch_acc, epoch)
            else:
                writer.add_scalar('data/valloss', epoch_loss, epoch)
                writer.add_scalar('data/valacc', epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

    writer.close()  # 关闭SummaryWriter
    return model  # 返回训练好的模型


if __name__ == '__main__':
    image_size = 64  # 图像缩放大小
    crop_size = 48  # 图像裁剪大小，即网络输入大小
    nclass = 4  # 分类类别数量
    model = ThreeConvNet(nclass)  # 实例化模型
    data_dir = './data'  # 数据存放目录

    # 如果不存在模型保存目录则创建之
    if not os.path.exists('models'):
        os.mkdir('models')

    # 判断是否有可用的GPU，如果有则使用，否则使用CPU
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()  # 将模型转移到GPU上，这可以显著加快模型的训练速度

    # 数据预处理步骤
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(crop_size),  # 随机裁剪到固定大小，增加模型训练时看到的图像多样性
            transforms.RandomHorizontalFlip(),  # 随机水平翻转，进一步增加数据多样性
            transforms.ToTensor(),  # 转换为Tensor
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化处理，加快模型训练的收敛速度
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size),  # 缩放图片
            transforms.CenterCrop(crop_size),  # 中心裁剪
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    # 使用ImageFolder读取图片，并应用预处理
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x]) for x in ['train', 'val']}

    # 创建DataLoader加载数据
    dataloaders = {x: DataLoader(image_datasets[x],
                                 batch_size=64,  # 批次大小
                                 shuffle=True,  # 是否打乱顺序，有助于减少模型对数据顺序的依赖
                                 num_workers=4)  # 多进程加载的进程数，提高数据加载效率
                   for x in ['train', 'val']}
    # 获取各部分数据集的大小
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # 定义损失函数、优化器以及学习率调度器
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失
    optimizer_ft = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)  # 使用带动量的SGD优化器，动量可以帮助加速 SGD 在相关方向上的优化
    step_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100,
                                            gamma=0.1)  # 学习率每100个epoch衰减为原来的0.1倍，有助于模型在损失平台期找到更深的局部最小值

    # 开始训练模型，这里设置为300个epoch
    model = train_model(model=model,
                        criterion=criterion,
                        optimizer=optimizer_ft,
                        scheduler=step_lr_scheduler,
                        num_epochs=300)

    # 训练完成后保存模型参数
    torch.save(model.state_dict(), 'models/model.pt')
