#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import tensorflow as tf

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_category, read_vocab, get_words, pad_sequences


base_path = 'data/cnews'
vocab_path = os.path.join(base_path, 'cnews.vocab.txt')

save_path = 'checkpoints/textcnn'
save_path = os.path.join(save_path, 'best_validation')  # 最佳验证结果保存路径


class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()

        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_path)

        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, content):
        data = [self.word_to_id[x] for x in get_words(content) if self.word_to_id.get(x)]

        feed_dict = {
            self.model.input_x: pad_sequences([data]),
            self.model.keep_prob: 1.0,
            self.model.is_train: False
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    cnn_model = CnnModel()
    test_demo = ['三星ST550以全新的拍摄方式超越了以往任何一款数码相机',
                 '热火vs骑士前瞻：皇帝回乡二番战 东部次席唾手可得新浪体育讯北京时间3月30日7:00']
    for i in test_demo:
        print(cnn_model.predict(i))
