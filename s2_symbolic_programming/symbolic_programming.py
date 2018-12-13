"""
tensorflow 符号式编程
"""

import tensorflow as tf

# 创建常量
a = tf.constant(2)
b = tf.constant(3)

# 乘法
c = tf.multiply(a, b)

# 加法
d = tf.add(c, 1)

with tf.Session() as sess:
    print(sess.run(d))
