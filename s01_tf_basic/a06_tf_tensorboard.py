"""
TensorBoard

# 用tf保存图的信息到日志中
    tf.summary.FileWriter("日志保存路径", sess.graph)

# 启动tensorboard，用tensorboard读取并展示日志
    tensorboard --logdir="日志所在路径"

# name_scope 命名空间

游乐场：http://playground.tensorflow.org

"""
import tensorflow as tf

# 构造graph (用一个线性方程的例子 y=W*x+b)

# 权重
W = tf.Variable(2.0, dtype=tf.float32, name="Weight")  # 此处设置的名字会显示在tensorboard的图像上
# 偏差
b = tf.Variable(1.0, dtype=tf.float32, name="Bias")
# 输入
x = tf.placeholder(dtype=tf.float32, name="Input")

# 输出的命名空间
with tf.name_scope(name="Output"):
    y = W * x + b

# 定义保存日志的路径
path = "./log"

# 创建用于初始化所有变量(Variable)的操作
init = tf.global_variables_initializer()

# 创建session
with tf.Session() as sess:
    sess.run(init)  # 初始化变量
    #
    writer = tf.summary.FileWriter(path, sess.graph)
    result = sess.run(y, {x: 3.0})
    print("y = {}".format(result))
