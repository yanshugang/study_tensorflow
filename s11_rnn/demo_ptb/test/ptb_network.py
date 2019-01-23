import tensorflow as tf

# 词向量层（word embedding）
VOCAB_SIZE = "词汇表的大小"
EMB_SIZE = "词向量的维度"
input_data = "输入的数据"

embedding = tf.get_variable(name="embedding",
                            shape=[VOCAB_SIZE, EMB_SIZE])

# 输出的矩阵比输入数据多一个维度，新增维度的大小是EMR_SIZE。
# 语言模型中，一般input_data的维度是batch_size * num_steps,
# 而输出的input_embedding的维度是batch_size * num_steps * EMR_SIZE
imput_embedding = tf.nn.embedding_lookup(embedding, input_data)

# softmax层
# 将循环神经网络的输出转化为一个单词表中每个单词的输出概率。

# 步骤一：使用一个线性映射，将循环网络的输出映射为一个维度与词汇表大小相同的向量。
HIDDEN_SIZE = "循环神经网络的隐藏状态维度"
weight = tf.get_variable(name="weight",
                         shape=[HIDDEN_SIZE, VOCAB_SIZE])
bias = tf.get_variable(name="bias",
                       shape=[VOCAB_SIZE])
# 计算线性映射，output是RNN的输出，其维度为[batch_size * num_steps, HIDDEN_SIZE]
output = ""
logits = tf.nn.bias_add(tf.matmul(output, weight), bias)

# 步骤二：调用softmax方法将logits转化为和为1的概率。在VOCAB_SIZE个可能的类别中决定这一步最可能输出的单词。
probs = tf.nn.softmax(logits=logits)
# labels是一个大小为[batch_size * num_steps]的一维数组，它包含每个位置正确的单词编号
# logits的维度是[batch_size * num_steps, HIDDEN_SIZE]
# loss的维度与labels相同，代表每个位置上的log perplexity
# todo: targets是什么？？？
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(targets, [-1]),
                                                      logits=logits)
