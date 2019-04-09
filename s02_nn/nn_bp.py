"""
使用反向传播算法训练神经网络
"""

import tensorflow as tf

w1 = tf.Variable(tf.random_normal(shape=[2, 3],
                                  stddev=1,
                                  seed=1))
w2 = tf.Variable(tf.random_normal(shape=[3, 1],
                                  stddev=1,
                                  seed=1))

# 定义placeholder作为存放输入数据的地方
x = tf.placeholder(dtype=tf.float32,
                   shape=(1, 2),
                   name='input')
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数：刻画当前的预测值和真实答案间的差距
y_ = tf.sigmoid(y)

cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
    +
    (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0))
)

# 定义学习率
learning_rate = 0.001

# 定义反向传播算法来优化神经网络中的参数
train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    res = sess.run(y, feed_dict={x: [[0.7, 0.9]]})
    print(res)
