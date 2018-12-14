"""
用梯度下降的优化方法来快速解决线性回归问题
"""

# 构建数据
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

points_num = 100
vectors = []

# 用numpy的正态随机分布函数生成100个点
# 这些点的（x, y）坐标值 对应线性方程y = 0.1x + 0.2
# 权重(weight)0.1， 偏差(bias)0.2

for i in range(points_num):
    x1 = np.random.normal(0.0, 0.66)
    y1 = 0.1 * x1 + 0.2 + np.random.normal(0.0, 0.04)
    vectors.append([x1, y1])

x_data = [v[0] for v in vectors]
y_data = [v[1] for v in vectors]

# 图像1：展示所有的随机数据
plt.plot(x_data, y_data, "r*", label="Original data")  # 红色星型
plt.title("LR_using_GD")
plt.legend()
plt.show()

# 构建线性回归模型
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # 初始化weight
b = tf.Variable(tf.zeros([1]))  # 初始化bias
y = W * x_data + b  # 使用模型计算出y

# 定义损失函数: loss function
# 对Tensor的所有维度计算((y - y_data)^2)之和/N
loss = tf.reduce_mean(tf.square(y - y_data))

# 使用梯度下降的优化器来优化loss function
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 设置学习率：0.5
train = optimizer.minimize(loss)

# 创建会话
sess = tf.Session()

# 初始化数据流图中的所有变量
init = tf.global_variables_initializer()
sess.run(init)

# 训练20步
for step in range(100):
    # 优化每一步
    sess.run(train)
    # 打印出每一步的损失，权重和偏差
    print("step=%s, loss=%s, weight=%s, bias=%s" % (step, sess.run(loss), sess.run(W), sess.run(b)))

# 图像2: 绘制所有的点并且绘制出最佳拟合的直线
plt.plot(x_data, y_data, "r*", label="Original data")  # 红色星型
plt.title("LR_using_GD")
plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label="fitted line")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()

plt.close()
