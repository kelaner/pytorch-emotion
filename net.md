```
ThreeConvNet(
  (conv1): Conv2d(3, 12, kernel_size=(3, 3), stride=(2, 2))
  (bn1): BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv2): Conv2d(12, 24, kernel_size=(3, 3), stride=(2, 2))
  (bn2): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv3): Conv2d(24, 48, kernel_size=(3, 3), stride=(2, 2))
  (bn3): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc1): Linear(in_features=1200, out_features=1200, bias=True)
  (fc2): Linear(in_features=1200, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=4, bias=True)
)
```

模型包含三个卷积层和三个全连接层，每个卷积层后面都跟着一个批量归一化层。

1. `conv1`: 第一个卷积层，输入通道数为3（RGB三通道），输出通道数为12，卷积核大小为3x3，步长为2x2。

2. `bn1`: 第一个批量归一化层，对应于第一个卷积层，使得网络在各层的输入数据保持相同的分布，有利于网络的训练。

3. `conv2`: 第二个卷积层，输入通道数为12，输出通道数为24，卷积核大小为3x3，步长为2x2。

4. `bn2`: 第二个批量归一化层，对应于第二个卷积层。

5. `conv3`: 第三个卷积层，输入通道数为24，输出通道数为48，卷积核大小为3x3，步长为2x2。

6. `bn3`: 第三个批量归一化层，对应于第三个卷积层。

7. `fc1`: 第一个全连接层，输入特征数为1200，输出特征数也为1200。

8. `fc2`: 第二个全连接层，输入特征数为1200，输出特征数为128。

9. `fc3`: 第三个全连接层，输入特征数为128，输出特征数为4。这是最后一层，输出的特征数通常等于分类任务的类别数。

使用时，图片会首先通过三个卷积层和批量归一化层进行特征提取，然后通过三个全连接层进行特征转换和分类。
