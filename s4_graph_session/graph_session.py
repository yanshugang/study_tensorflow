"""
图和会话
"""

# 数据流图：-> tf.Graph
#     数据 data
#     流 flow
#     图 graph


# session 会话

# tensorflow采用客户端、服务端架构，客户端使用session.run()命令向服务端发起执行，session的作用就是让静态的图运行起来。

# tensorflow程序的流程：
#     定义算法的graph结构
#     使用session去执行


import tensorflow as tf

c = tf.constant(value=([[1, 2], [3, 4]]), dtype=tf.int64, name="const_1")
print(c)  # Tensor("const_1:0", shape=(2, 2), dtype=int64)

sess = tf.Session()
print(sess)  # <tensorflow.python.client.session.Session object at 0x11d80a8d0>

result = sess.run(c)
print(result)

if c.graph is tf.get_default_graph():
    print("c graph is default graph")
