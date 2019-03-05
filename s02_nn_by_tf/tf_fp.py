# -*- coding: utf-8 -*-

# @Author: ysg
# @Contact: yanshugang11@163.com
# @Time: 2019/3/5 下午5:16

"""
向前传播 forward-propagation
可以使用矩阵乘法表示：tf.matmul()
"""
# TODO:
import tensorflow as tf

# 声明一个2*3的矩阵
weight = tf.Variable(tf.random_normal([2, 3], stddev=2))
# 声明
biases = tf.Variable(tf.zeros(shape=[3]))
