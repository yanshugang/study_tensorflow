"""
使用cifar-10数据集, 实现神经元多分类模型
使用多个神经元做输出
"""
import os

import numpy as np
import pickle
import tensorflow as tf

CIFAR_DIR = "./cifar-10-batches-py"


def load_data(file_name):
    """
    read data from data file. (use cifar-10 dataset)
    """
    with open(file_name, 'rb') as fr:
        data = pickle.loads(fr.read(), encoding='iso-8859-1')
        return data['data'], data["labels"]


# tensorflow有封装Dataset
class CifarData:
    """
    cifar数据处理
    """

    def __init__(self, file_names, need_shuffle):
        all_data = []
        all_labels = []

        for file_name in file_names:
            data, labels = load_data(file_name)

            # 包含全部数据，不做filter, 10个类
            all_data.append(data)
            all_labels.append(labels)

        self._data = np.vstack(all_data)  # 纵向合成矩阵
        self._data = self._data / 127.5 - 1  # 归一化，解决梯度消失¬
        self._labels = np.hstack(all_labels)  # 横向合成矩阵

        # print(self._data.shape)
        # print(self._labels.shape)

        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0

        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        # TODO: 混洗
        p = np.random.permutation(self._num_examples)
        # print("shuffle: %s" % p)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        """
        return batch_size examples as a batch.
        """
        end_indicator = self._indicator + batch_size

        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("have no more examples")

        if end_indicator > self._num_examples:
            raise Exception("batch size is larger than all examples")

        batch_data = self._data[self._indicator: end_indicator]
        batch_labels = self._labels[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels


train_file_names = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 6)]
test_file_name = [os.path.join(CIFAR_DIR, 'test_batch')]

train_data = CifarData(train_file_names, True)
test_data = CifarData(test_file_name, False)

# 测试CifarData
# batch_data, batch_labels = train_data.next_batch(10)
# print(batch_data)
# print(batch_labels)

# tf.placeholder 占位符
x = tf.placeholder(tf.float32, [None, 3072])
y = tf.placeholder(tf.int64, [None])

# tf.get_variable 获取变量
# w: (3071, 10)
# initializer 初始化
w = tf.get_variable(name='w',
                    shape=[x.get_shape()[-1], 10],
                    initializer=tf.random_normal_initializer(0, 1),  # 使用正态分布初始化w
                    )
# b: (10,)
b = tf.get_variable(name='b',
                    shape=[10],
                    initializer=tf.constant_initializer(0.0),
                    )

# [None, 3072] * [3072, 10] = [None, 10]   (tf.matmul 矩阵乘法)
y_ = tf.matmul(x, w) + b

"""
# 平方差损失函数
p_y = tf.nn.softmax(y_)
# 对y进行one hot编码
y_one_hot = tf.one_hot(y, 10, dtype=tf.float32)
# 损失函数
loss = tf.reduce_mean(tf.square(y_one_hot - p_y))
"""

# 交叉熵损失函数
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

"""
# 使用sigmoid变成概率值, 得到y=1的概率值. [None, 1]
p_y_1 = tf.nn.sigmoid(y_)

# 先对y做reshape
y_reshaped = tf.reshape(y, (-1, 1))
y_reshaped_float = tf.cast(y_reshaped, tf.float32)
# 损失函数
loss = tf.reduce_mean(tf.square(y_reshaped_float - p_y_1))
"""

# 预测值
predict = tf.argmax(y_, 1)

# 准确率
correct_prediction = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

with tf.name_scope("train_op"):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()
batch_size = 20
train_steps = 100000
test_steps = 100

with tf.Session() as sess:
    sess.run(fetches=init)
    for i in range(train_steps):
        batch_data, batch_labels = train_data.next_batch(batch_size)

        loss_val, acc_val, _ = sess.run([loss, accuracy, train_op],
                                        feed_dict={x: batch_data, y: batch_labels})
        if (i + 1) % 500 == 0:
            print("[Train] step: %s, loss: %4.5f, acc: %4.5f" % (i + 1, loss_val, acc_val))

        if (i + 1) % 5000 == 0:
            test_data = CifarData(test_file_name, False)
            all_test_acc_val = []
            for j in range(test_steps):
                test_batch_data, test_batch_labels = test_data.next_batch(batch_size)
                test_acc_val = sess.run([accuracy],
                                        feed_dict={x: test_batch_data, y: test_batch_labels})
                all_test_acc_val.append(test_acc_val)

            test_acc = np.mean(all_test_acc_val)
            print("==============[Test] Step: %s, acc: %4.5f" % (i + 1, test_acc))
