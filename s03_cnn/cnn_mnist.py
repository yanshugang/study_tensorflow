"""
慕课视频: CNN-MNIST
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data", one_hot=True,
                                  source_url="http://yann.lecun.com/exdb/mnist/")  # 55000张，28*28像素

# None表示张量(Tensor)的第一个维度可以是任何长度
# 除以255是为了做归一化(Normalization)，把灰度值从[0, 255]变成[0, 1]区间
# 归一话可以让之后的优化器(optimizer)更快更好地找到误差最小值
input_x = tf.placeholder(dtype=tf.float32, shape=[None, 28 * 28]) / 255  # 输入。是一个28*28的图片。
output_y = tf.placeholder(dtype=tf.int32, shape=[None, 10])  # 输出：10个数字的预测

# -1表示自动推导维度大小。让计算机根据其他维度的值和总的元素大小来推导出-1的地方的维度应该是多少
input_x_images = tf.reshape(input_x, [-1, 28, 28, 1])  # 改变形状之后的输入

# 测试数据，从测试数据集里选取3000个手写数字的图片和对应标签
test_x = mnist.test.images[:3000]  # 图片
test_y = mnist.test.labels[:3000]  # 标签

# 构建卷积神经网络
# 卷积层
conv_1 = tf.layers.conv2d(inputs=input_x_images,  # 形状[28, 28, 1]
                          filters=32,  # 32个过滤器，输出的深度(depth=32)
                          kernel_size=[5, 5],  # 过滤器的二维平面大小 5*5
                          strides=1,  # 步长
                          padding='same',  # 补零方案: same表示输出的大小不变，因此需要在外围补零两圈
                          activation=tf.nn.relu  # 激活函数：relu
                          )  # 输出形状[28, 28, 32]

# 池化层(pooling): 亚采样
pool_1 = tf.layers.max_pooling2d(inputs=conv_1,  # 形状[28, 28, 32]
                                 pool_size=[2, 2],  # 过滤器在二维的大小是2*2
                                 strides=2,  # 步长2
                                 name="first_pooling"
                                 )  # 输出形状[14, 14, 32]

# 卷积层
conv_2 = tf.layers.conv2d(inputs=pool_1,  # 形状：14*14*32
                          filters=64,  # 64个过滤器，输出的深度(depth=64)
                          kernel_size=[5, 5],  # 过滤器的二维平面大小
                          strides=1,  # 步长
                          padding='same',  # 补零方案:same, 表示输出的大小不变，因此需要在外围补零两圈
                          activation=tf.nn.relu  # 激活函数：relu
                          )  # 输出形状[14, 14, 64]

# 池化层: 亚采样
pool_2 = tf.layers.max_pooling2d(inputs=conv_2,  # 形状[14, 14, 64]
                                 pool_size=[2, 2],  # 过滤器在二维的大小是2*2
                                 strides=2,  # 步长2
                                 )  # 输出形状[7, 7, 64]

# 平坦化 flat
flat = tf.reshape(pool_2, shape=[-1, 7 * 7 * 64])  # 形状：7*7*64

# 全连接层: 1024个神经元的全连接层
dense = tf.layers.dense(inputs=flat,
                        units=1024,
                        activation=tf.nn.relu
                        )

# dropout: 丢弃50%，即rate=0.5
dropout = tf.layers.dropout(inputs=dense,
                            rate=0.5
                            )

# 全连接层: 构建10个神经元的全连接层，这里不用激活函数来做非线性化了
logits = tf.layers.dense(inputs=dropout,
                         units=10
                         )  # 输出形状：1*1*10

# 计算误差（计算cross entropy（交叉熵），再用sotfmax计算百分比的概率）
loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y,
                                       logits=logits
                                       )

# 用Adam优化器来最小化误差，学习率0.001
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# 精度。计算预测值和实际标签的匹配程度。返回(accuracy, update_op), 会创建两个局部变量。
accuracy = tf.metrics.accuracy(labels=tf.argmax(output_y, axis=1),
                               predictions=tf.argmax(logits, axis=1),
                               )[1]

# 创建会话
sess = tf.Session()
# 初始化变量：要初始化全局和局部两种变量
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)

for i in range(2000):
    batch = mnist.train.next_batch(50)  # 从训练数据集里取下一个'50个样本'
    train_loss, train_op_ = sess.run([loss, train_op], {input_x: batch[0], output_y: batch[1]})

    if i % 100 == 0:
        test_accuracy = sess.run(accuracy, {input_x: test_x, output_y: test_y})
        print("step={}, train loss={}, test accoutacy={}".format(i, train_loss, test_accuracy))

# 测试： 打印20个预测值和真实值的对
test_output = sess.run(logits, {input_x: test_x[:20]})
inferred_y = np.argmax(test_output, 1)

print(inferred_y, '推测的数字')  # 推测的数字
print(np.argmax(test_y[:20], 1), '真实的数字')  # 真实的数字

# 关闭会话
sess.close()
