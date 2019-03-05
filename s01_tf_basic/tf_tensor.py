# -*- coding: utf-8 -*-

# @Author: ysg
# @Contact: yanshugang11@163.com
# @Time: 2019/3/5 下午2:44
"""
tensor 张量
"""
import tensorflow as tf

a = tf.constant([1, 2], name="a", dtype=tf.int32)
b = tf.constant([2, 3], name="b", dtype=tf.int32)

result = a + b
