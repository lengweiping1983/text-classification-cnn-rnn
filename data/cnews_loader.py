#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os
from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr
import thulac


if sys.version_info[0] > 2:
    is_py3 = True
else:
    # reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


stopwords_path = 'data/stopwords.txt'  # 停用词表路径
need_segmentation = True               # 是否需要分词
if need_segmentation:
    thu = thulac.thulac(seg_only=True)


def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def get_words(content):
    """获取词汇"""
    if need_segmentation:
        text = thu.cut(content)
        words = [x[0].strip() for x in text if x[0].strip()]
    else:
        words = list(content)
    return words


def read_stopword(stopwords_path):
    """读取停用词"""
    if stopwords_path:
        with open_file(stopwords_path) as f:
            stopwords = [native_content(_.strip()) for _ in f.readlines()]
            stopwords = dict(zip(stopwords, range(len(stopwords))))
            return stopwords
    else:
        return None


def filter_stopword(words, stopwords):
    """过滤停用词"""
    return [x for x in words if not stopwords.get(x)]


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    if os.path.exists(filename + '.seg.txt') and os.path.exists(filename + '.label.txt'):
        with open_file(filename + '.seg.txt') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                content = []
                for word in line.split(" "):
                    content.append(word)
                contents.append(content)
        with open_file(filename + '.label.txt') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                labels.append(line)
    else:
        stopwords = read_stopword(stopwords_path)

        with open_file(filename) as f:
            for line in f:
                try:
                    label, content = line.strip().split('\t')
                    if content:
                        words = get_words(native_content(content))
                        if stopwords:
                            words = filter_stopword(words, stopwords)
                        if words:
                            contents.append(words)
                            labels.append(native_content(label))
                except:
                    pass
        if contents and labels:
            with open_file(filename + '.seg.txt', mode='w') as f:
                for content in contents:
                    f.write(" ".join(content) + '\n')
            open_file(filename + '.label.txt', mode='w').write('\n'.join(labels) + '\n')
    return contents, labels


def merge_stopword(source_path1, source_path2, stopwords_path):
    """合并停用词"""
    stopwords1 = read_stopword(source_path1)
    stopwords2 = read_stopword(source_path2)
    stopwords = [x.strip() for x in stopwords1.keys() if x.strip()]
    for x in stopwords2.keys():
        if x.strip():
            if not stopwords1.get(x.strip()):
                stopwords.append(x.strip())
    open_file(stopwords_path, mode='w').write('\n'.join(stopwords) + '\n')


def build_vocab(train_path, vocab_path, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    train_data, _ = read_file(train_path)

    all_data = []
    for content in train_data:
        all_data.extend(content)

    counter = Counter(all_data)
    min_vocab_size = min(len(counter.items()), vocab_size)
    count_pairs = counter.most_common(min_vocab_size)
    words, _ = list(zip(*count_pairs))
    open_file(vocab_path, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_path):
    """读取词汇表"""
    with open_file(vocab_path) as f:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in f.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """读取分类目录，固定"""
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

    categories = [native_content(x) for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if word_to_id.get(x)])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def pad_sequences(data_id, max_length=600):
    pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    return pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


if __name__ == '__main__':
    merge_stopword('stopwords1.txt', 'stopwords2.txt', 'stopwords.txt')
