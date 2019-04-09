"""
预处理
将文本转化为模型可以读入的单词序列
"""

import codecs
import collections
from operator import itemgetter


def clear_data(input_file, clearn_file):
    """

    :param input_file:
    :param output_file:
    :return:
    """
    # 统计词频
    counter = collections.Counter()
    with codecs.open(input_file, "r", "utf-8") as f:
        for line in f:
            for word in line.strip().split():
                counter[word] += 1

    # 按词频顺序对单词进行排序
    sorted_word_to_cnt = sorted(counter.items(),
                                key=itemgetter(1),
                                reverse=True)
    sorted_words = [x[0] for x in sorted_word_to_cnt]

    # 稍后我们需要在文本换行出加入句子结束符"<eos>", 这里预先将其加入词汇表
    sorted_words = ["<unk>", "<sos>", "<eos>"] + sorted_words

    if len(sorted_words) > 10000:
        sorted_words = sorted_words[:10000]

    with codecs.open(clearn_file, "w", "utf-8") as file_output:
        for word in sorted_words:
            file_output.write(word + "\n")


def word2num(input_file, clearn_file, output_file):
    """
    将单词转换为编号
    """
    RAW_DATA = "./data/ptb.train.txt"
    VOCAB = "ptb.vocab"
    OUTPUT_DATA = "ptb.train"

    # 建立词汇到编号的映射（单词编号即为行号）
    with codecs.open(clearn_file, "r", "utf-8") as f_vocab:
        vocab = [w.strip() for w in f_vocab.readlines()]
    word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

    # 如果出现了被删除的低频词，则替换为<unk>
    def get_id(word):
        return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]

    fin = codecs.open(input_file, "r", "utf-8")
    fout = codecs.open(output_file, "w", "utf-8")
    for line in fin:
        words = line.strip().split() + ["<eos>"]
        out_line = " ".join([str(get_id(w)) for w in words]) + "\n"
        fout.write(out_line)
    fin.close()
    fout.close()


def main():
    input_file = "./source_data/ptb.valid.txt"
    clearn_file = "./valid.clearn"
    output_file = "./used_data/ptb.valid"

    clear_data(input_file, clearn_file)
    word2num(input_file, clearn_file, output_file)


if __name__ == '__main__':
    main()
