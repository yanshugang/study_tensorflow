# -*- coding: utf-8 -*-

# @Author: ysg
# @Contact: yanshugang11@163.com
# @Time: 2019/3/12 下午4:56

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST数据集相关的常数
INPUT_NODE = 784  # 输入层的节点数。对于MNIST数据集，这个就等于图片的像素。
OUTPUT_NODE = 10  # 输出层的节点数。这个等于类别的数目。因为在MNIST数据集中需要区分的是0~9这10个数字，所以输出层的节点数为10。

# 配置神经网络参数
LAYER1_NODE = 500  # 隐藏层节点数。这里使用只有一个隐藏层的网络结构
BATCH_SIZE = 100  # 一个训练batch中的训练数据个数。TODO：数字越小时，训练过程越接近随机梯度下降；数字越大时，训练越接近梯度下降；
LEARNING_RATE_BASE = 0.8  # 基础的学习率
LEARING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # TODO：描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # TODO：滑动平均衰减率


# 一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果。在这里定义了一个使用ReLU激活函数的三层全连接神经网络。
# 通过加入隐藏层实现了多层网络结构，通过ReLU激活函数实现了去线性化。
# 在这个函数中也支持传入用于计算参数平均值的类，这样方便在测试时使用滑动平均模型。
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 当没有提供滑动平均类时，直接使用参数当前的取值。
    if avg_class == None:
        # 计算隐藏层的前向传播结果，这里使用了ReLU激活函数。
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)


# 下载数据
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True, source_url="http://yann.lecun.com/exdb/mnist/")
