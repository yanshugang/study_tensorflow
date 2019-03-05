"""
tf基本结构:
    graph: 数据流图。[计算模型]
    tensor: 张量。[数据模型]
    session: 会话。[运行模型]

# 结点（operation 操作）
"""

import tensorflow as tf

# graph
# 定义变量
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")

# 获取当前默认计算图
print(tf.get_default_graph())

# 定义一个新的计算图
g1 = tf.Graph()
# 指定运行设备
with g1.device("/cpu:0"):
    result = a + b
    print(result)


