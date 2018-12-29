"""
使用cifar-10数据集
"""
import pickle
import tensorflow as tf

CIFAR_DIR = "./cifar-10-batches-py"


def load_data(file_name):
    """
    read data from data file. (use cifar-10 dataset)
    :return:
    """
    with open(file_name, 'rb') as fr:
        data = pickle.loads(fr.read(), encoding='iso-8859-1')
        return data['data'], data["labels"]


x = tf.placeholder(tf.float32, [None, 3072])
y = tf.placeholder(tf.int64, [None])

# TODO: tf.get_variable
# w: (3071, 1)
w = tf.get_variable('w', [x.get_shape()[-1], 1],
                    initializer=tf.random_normal_initializer(0, 1))
# b: (1,)
b = tf.get_variable('b', [1],
                    initializer=tf.constant_initializer(0.0))

# [None, 3072] * [3072, 1] = [None, 1]   (tf.matmul 矩阵乘法)
y_ = tf.matmul(x, w) + b

# 使用sigmoid变成概率值, 得到y=1的概率值. [None, 1]
p_y_1 = tf.nn.sigmoid(y_)

# 先对y做reshape, todo: -1是什么意思
y_reshaped = tf.reshape(y, (-1, 1))
y_reshaped_float = tf.cast(y_reshaped, tf.float32)
# 损失函数
loss = tf.reduce_mean(tf.square(y_reshaped_float - p_y_1))

# 预测值
predict = p_y_1 > 0.5

# 准确率
correct_prediction = tf.equal(tf.cast(predict, tf.int64), y_reshaped)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

with tf.name_scope("train_op"):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
