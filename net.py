import torch.nn as nn
import torch.nn.functional as F


# 定义一个三层卷积神经网络ThreeConvNet
class ThreeConvNet(nn.Module):
    # 初始化函数，nclass代表最终分类的类别数
    def __init__(self, nclass):
        super(ThreeConvNet, self).__init__()
        # 第一层卷积层，输入通道数3（彩色图片），输出通道数12，卷积核大小3x3，步长为2
        self.conv1 = nn.Conv2d(3, 12, 3, 2)
        # 第一层批归一化层，对应于12个输出通道
        self.bn1 = nn.BatchNorm2d(12)
        # 第二层卷积层，输入通道数12，输出通道数24，卷积核大小3x3，步长为2
        self.conv2 = nn.Conv2d(12, 24, 3, 2)
        # 第二层批归一化层，对应于24个输出通道
        self.bn2 = nn.BatchNorm2d(24)
        # 第三层卷积层，输入通道数24，输出通道数48，卷积核大小3x3，步长为2
        self.conv3 = nn.Conv2d(24, 48, 3, 2)
        # 第三层批归一化层，对应于48个输出通道
        self.bn3 = nn.BatchNorm2d(48)
        # 第一层全连接层，输入特征维度为48*5*5，输出特征维度为1200
        self.fc1 = nn.Linear(48 * 5 * 5, 1200)
        # 第二层全连接层，输入特征维度为1200，输出特征维度为128
        self.fc2 = nn.Linear(1200, 128)
        # 第三层全连接层，输入特征维度为128，输出特征维度为分类数nclass
        self.fc3 = nn.Linear(128, nclass)

    # 前向传播函数定义，x为输入的数据
    def forward(self, x):
        # 通过第一层卷积层后，使用ReLU激活函数
        x = F.relu(self.bn1(self.conv1(x)))
        # 通过第二层卷积层后，使用ReLU激活函数
        x = F.relu(self.bn2(self.conv2(x)))
        # 通过第三层卷积层后，使用ReLU激活函数
        x = F.relu(self.bn3(self.conv3(x)))
        # 将卷积层输出的多维数据展平成一维数据
        x = x.view(-1, 48 * 5 * 5)
        # 通过第一层全连接层后，使用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 通过第二层全连接层后，使用ReLU激活函数
        x = F.relu(self.fc2(x))
        # 通过第三层全连接层得到最终的输出
        x = self.fc3(x)
        return x


# 测试代码
if __name__ == '__main__':
    import torch

    # 创建一个随机数据tensor模拟输入图像，大小为1x3x48x48
    x = torch.randn(1, 3, 48, 48)
    # 实例化网络模型，假设有4个分类
    model = ThreeConvNet(4)
    # 将输入数据传入模型获取输出
    y = model(x)
    # 打印模型结构
    print(model)
