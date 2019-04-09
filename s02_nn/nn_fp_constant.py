# -*- coding: utf-8 -*-

# @Author: ysg
# @Contact: yanshugang11@163.com
# @Time: 2019/3/5 下午5:16

"""
向前传播 forward-propagation
可以使用矩阵乘法表示：tf.matmul()
"""
import tensorflow as tf

# 声明w1、w2两个变量，并通过seed参数设定随机种子，保证每次运行的结果一样。
w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))  # todo: stddev, seed 分别指什么
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))

# 特征向量（使用常量表示）
x = tf.constant([[0.7, 0.9]])

# 向前传播
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    # 初始化全部变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(y))
