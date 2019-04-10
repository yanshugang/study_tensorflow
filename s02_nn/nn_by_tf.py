"""
一个完整的神经网络，解决二分类问题，使用反向传播算法训练神经网络。
"""

import tensorflow as tf
import numpy as np

# 定义训练batch的大小
batch_size = 8

# 定义参数
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='x_input')
y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='y_input')

# 定义神经网络前向传播的过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播的算法
y = tf.sigmoid(y)

# 交叉熵
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
    +
    (1 - y) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0))
)

# 定义学习率
learning_rate = 0.001

# 定义反向传播算法来优化神经网络中的参数
train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
rdm = np.random.RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# 创建一个会话，用来运行TF程序
with tf.Session() as sess:
    # 初始化全部变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print("=======初始值=======")
    print(sess.run(w1))
    print(sess.run(w2))
    print("===================")

    # 设定训练的轮数
    STEPS = 50000

    for i in range(STEPS):
        # 每次选取batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        if i % 1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(fetches=cross_entropy,
                                           feed_dict={x: X, y_: Y})

            print("After %s training steps, cross entropy on all data is %s" % (i, total_cross_entropy))

    print("=======result=======")
    print(sess.run(w1))
    print(sess.run(w2))
    print("====================")
