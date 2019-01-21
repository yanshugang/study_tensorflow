"""
将文本转化为模型可以读入的单词序列
"""

import codecs
import collections
from operator import itemgetter

# 训练集数据文件
RAW_DATA = "./data/ptb_train.txt"
# 输出的词汇表文件

# 统计词频
counter = collections.Counter()
with codecs.open(RAW_DATA, "r", "utf-8") as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1

# 按词频顺序对单词进行排序
sorted_word_to_cnt = sorted(counter.items(),
                            key=itemgetter(1),
                            reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]

# 稍后我们需要在文本换行出加入句子结束符"<eos>", 这里预先将其加入词汇表
sorted_words = ["<eos>"] + sorted_words

#
