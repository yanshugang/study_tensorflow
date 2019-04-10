# -*- coding: utf-8 -*-

# @Author: ysg
# @Contact: yanshugang11@163.com
# @Time: 2019/3/7 下午5:04

"""
向前传播
"""
import tensorflow as tf

# 声明w1、w2两个变量，并通过seed参数设定随机种子，保证每次运行的结果一样。
w1 = tf.Variable(tf.random_normal(shape=(2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal(shape=(3, 1), stddev=1, seed=1))

# # 特征向量（使用常量表示）
# x = tf.constant([[0.7, 0.9]])

# 特征向量（使用placeholder表示）
x = tf.placeholder(tf.float32, shape=(1, 2), name="inpuut")

# 向前传播
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    # 初始化全部变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(y, feed_dict={x: [[0.7, 0.9]]}))
