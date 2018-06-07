#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from rnn_model import TRNNConfig, TextRNN
from data.cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab

base_path = 'data/cnews'
train_path = os.path.join(base_path, 'cnews.train.txt')
test_path = os.path.join(base_path, 'cnews.test.txt')
val_path = os.path.join(base_path, 'cnews.val.txt')
vocab_path = os.path.join(base_path, 'cnews.vocab.txt')

tensorboard_path = 'tensorboard/textrnn'
save_path = 'checkpoints/textrnn'
save_path = os.path.join(save_path, 'best_validation')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(batch_x, batch_y, keep_prob, is_train=False):
    feed_dict = {
        model.input_x: batch_x,
        model.input_y: batch_y,
        model.keep_prob: keep_prob,
        model.is_train: is_train
    }
    return feed_dict


def evaluate(sess, x, y):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x)
    batch_eval = batch_iter(x, y, batch_size=128)
    total_loss = 0.0
    total_acc = 0.0
    for batch_x, batch_y in batch_eval:
        batch_len = len(batch_x)
        feed_dict = feed_data(batch_x, batch_y, keep_prob=1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_path)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("Loading training data...")
    # 载入训练集与验证集
    start_time = time.time()
    train_x, train_y = process_file(train_path, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    print("Loading validation data...")
    start_time = time.time()
    val_x, val_y = process_file(val_path, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0             # 总批次
    best_val_acc = 0.0          # 最佳验证集准确率
    last_improved = 0           # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    jump_flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(train_x, train_y, config.batch_size)
        for batch_x, batch_y in batch_train:
            feed_dict = feed_data(batch_x, batch_y, keep_prob=config.dropout_keep_prob, is_train=True)
            session.run(model.optim, feed_dict=feed_dict)  # 运行优化

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                train_loss, train_acc = session.run([model.loss, model.acc], feed_dict=feed_dict)
                val_loss, val_acc = evaluate(session, val_x, val_y)

                if val_acc > best_val_acc:
                    # 保存最好结果
                    best_val_acc = val_acc
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>8}, Train Loss: {1:>8.2}, Train Acc: {2:>8.2%},' \
                      + ' Val Loss: {3:>8.2}, Val Acc: {4:>8.2%}, Time: {5} {6}'
                print(msg.format(total_batch, train_loss, train_acc, val_loss, val_acc, time_dif, improved_str))

            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                jump_flag = True
                break  # 跳出循环
        if jump_flag:  # 同上
            break


def test():
    print("Loading test data...")
    start_time = time.time()
    test_x, test_y = process_file(test_path, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    test_loss, test_acc = evaluate(session, test_x, test_y)
    msg = 'Test Loss: {0:>8.2}, Test Acc: {1:>8.2%}'
    print(msg.format(test_loss, test_acc))

    batch_size = 128
    data_len = len(test_x)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(test_y, 1)
    y_pred_cls = np.zeros(shape=data_len, dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: test_x[start_id:end_id],
            model.keep_prob: 1.0,
            model.is_train: False
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_rnn.py [train / test]""")

    print('Configuring RNN model...')
    config = TRNNConfig()
    if not os.path.exists(vocab_path):  # 如果不存在词汇表，重建
        build_vocab(train_path, vocab_path, config.vocab_size)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_path)
    config.vocab_size = len(words)
    model = TextRNN(config)

    if sys.argv[1] == 'train':
        train()
    else:
        test()
