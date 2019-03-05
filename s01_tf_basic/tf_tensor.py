# -*- coding: utf-8 -*-

# @Author: ysg
# @Contact: yanshugang11@163.com
# @Time: 2019/3/5 下午2:44
"""
tensor 张量

张量的维度（秩）：rank\order
    scalar 标量
    vector 向量
    matrix 矩阵
    3D array 三阶张量

tensor属性：
    数据类型 dtype
    形状 shape

几种Tensor:
    constant 常量 -> tf.constant
    variable 变量 -> tf.Variable
    placeholder 占位符 -> tf.placeholder
    SparseTensor 稀疏张量 -> tf.sparse.SparseTensor
"""

import numpy as np
import tensorflow as tf

# constant
tensor_cons = tf.constant([1, 2, 3, 4])
a = tf.constant([1, 2], name="a", dtype=tf.int32)

# variable
tensor_var = tf.Variable([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)

# placeholder
x = tf.placeholder(dtype=tf.float32, shape=(3, 4))
y = tf.matmul(x, x)

with tf.Session() as sess:
    # print(sess.run(y))  # will fail because x was not fed.

    # 给占位符赋值
    rand_array = np.random.rand(3, 4)
    print(sess.run(y, feed_dict={x: rand_array}))  # 使用字典格式的赋值

# sparsetensor
tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
