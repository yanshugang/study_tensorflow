"""
通过变量实现神经网络的参数并实现向前传播的过程。
"""

import tensorflow as tf

# 声明变量
w1 = tf.Variable(tf.random_normal(shape=(2, 3),
                                  stddev=1,
                                  seed=1))  # stddev, seed 分别指什么
w2 = tf.Variable(tf.random_normal(shape=(3, 1),
                                  stddev=1,
                                  seed=1))

# 特征向量
x = tf.constant([[0.7, 0.9]])

# 向前传播
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()

# 初始化变量
sess.run(w1.initializer)
sess.run(w2.initializer)

# 输出
res = sess.run(y)
print(res)
sess.close()
