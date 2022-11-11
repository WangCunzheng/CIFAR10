# CIFAR10
该项目为课程作业

## Dataset
CIFAR-10 是一个更接近普适物体的彩色图像数据集。该数据集是由 Hinton 的学生Alex Krizhevsky
和 Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。数据集一共包含 10 个类别的 RGB 彩色图片。

<img src="figs/cifar10.png" width="700px">

每个图片的尺寸为 32 × 32 ，每个类别有 6000 个图像，数据集中一共有 50000 张训练图片和 10000 张测试图片。

## Net-ShouZheng
网络 ShouZheng 是由 Conv2d、 BatchNorm2d、 ReLU、 
Dropout、 MaxPool2d、 Linear以及 Flatten 层组成的简单网络。

相较于通常训练的简易网络，增加了 Dropout 层来增加网络鲁棒性。
