import tensorflow as tf

# 创建两个常量
const_1 = tf.constant(value=([[2, 2]]), name="const_1")  # 一行二列
const_2 = tf.constant(value=([[4], [4]]), name="const_2")  # 二行一列

# matmul 矩阵乘法
multiple = tf.matmul(const_1, const_2)

# 创建session
with tf.Session() as sess:
    result = sess.run(multiple)
    print(result)
