"""
RNN-LSTM文本分类
"""
import os

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


# 设置参数集合
def get_default_params():
    return tf.contrib.training.HParams(
        num_embedding_size=16,  #
        num_timesteps=50,  # 指定LSTM步长
        num_lstm_nodes=[32, 32],  # 指定LSTM每层的size
        num_lstm_layers=2,  # 指定LSTM层数，每层都是32个神经单元
        num_fc_nodes=32,  # todo:??
        batch_size=100,
        clip_lstm_grads=1.0,  # 设置梯度上限，控制LSTM梯度大小
        learning_rate=0.001,
        num_word_threshold=10,
    )


hps = get_default_params()

# 定义输入和输出文件
train_file = ""
val_file = ""
test_file = ""
vocab_file = ""
category_file = ""
output_file = ""

if not os.path.exists(output_file):
    os.mkdir(output_file)


class TextDataSet:
    def __init__(self, filename, vocab, category_vocab, num_timestrps):
        self._vocab = vocab
        self._category_vocab = category_vocab
