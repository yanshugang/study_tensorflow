# -*- coding: utf-8 -*-

# @Author: ysg
# @Contact: yanshugang11@163.com
# @Time: 2019/4/10 下午3:48

"""
定义神经网络的训练过程
"""
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from s03_mnist.mnist_best_practices import mnist_inference

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存的路径和文件名
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mpdel.ckpt"


def train(mnist):
    # 定义输入输出placeholder
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name="y-input")

    # 正则化函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    # 直接使用mnist_inferencr.py中定义的前向传播过程
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义滑动平均操作
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 定义损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    # 定义学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    # 定义训练步骤
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    # 初始化TensorFlow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 在训练过程中不再测试模型在验证数据上的表现，验证和测试的过程将会有一个独立的程序来完成。
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            # 每1000轮保存一次模型。
            if i % 1000 == 0:
                # 输出当前的训练情况。这里只输出了模型在当前训练batch上的损失函数大小。
                # 通过损失函数的大小可以大概了解训练的情况。
                # 在验证数据集上的正确率信息会有一个单独的程序来生成。
                print("After %d training step, loss on trianing batch is %g." % (step, loss_value))

                # 保存当前的模型，注意这里给出了global_step参数，这样可以让每个被保存模型的文件名末尾加上训练的轮数，
                # 比如"model。ckpt-1000"表示训练1000轮之后得到的模型。
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True, source_url="http://yann.lecun.com/exdb/mnist/")
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
