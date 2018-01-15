# -*-coding:utf-8-*-
"""
CBOW
1. Projection layer：将一句话拆分成多个样本，样本长度为embedding长度
3. Output layer：仿射变换，适配输出维度
4. Softmax or NCE
潘晓彤
2017.12.7

样本总量：sample_size
训练样本总量：train_set_size
one-hot长度(字典长度)：words_in_sentence，短文本补零处理
embedding长度：cfg.embedding_size
embedding窗口长度：cfg.embedding_window
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import numpy as np
import tensorflow as tf
from config import cfg

class CBOW():
    def __init__(self, sess, embed_file):
        self.input_samples = tf.placeholder(shape=[cfg.batch_size, cfg.embedding_window], dtype=tf.int32)
        self.label = tf.placeholder(shape=[cfg.batch_size], dtype=tf.int32)
        self.one_hot_label = tf.placeholder(shape=[cfg.batch_size, dictionary_size], dtype=tf.int32)
        self.embed_file = embed_file
        self.sess = sess

        with tf.variable_scope("cbow") as scope:
            # 特征矩阵：[dictionary_size, cfg.embedding_size]
            self.embed_weight = tf.get_variable(
                'embed_weight',
                shape=(dictionary_size, cfg.embedding_size),
                initializer=tf.random_normal_initializer(stddev=cfg.stddev),
                dtype='float64'
            )

            # 特征矩阵bias：dictionary_size
            embed_bias = tf.get_variable(
                'embed_bias',
                shape=(dictionary_size),
                initializer=tf.random_normal_initializer(stddev=cfg.stddev),
                dtype='float64'
            )

            # 样本embedding矩阵：[cfg.batch_size, cfg.embedding_window, cfg.embedding_size]
            embed_init = tf.nn.embedding_lookup(self.embed_weight, self.input_samples)

            # CBOW将上下文窗口中每个词的embedding相加
            proj_layer = tf.reduce_sum(embed_init, axis=1)

            # 从projection层到输出层权重矩阵：[cfg.embedding_size, dictionary_size]
            proj_output_weight = tf.get_variable(
                'proj_output_weight',
                shape=(cfg.embedding_size, dictionary_size),
                initializer=tf.random_normal_initializer(stddev=cfg.stddev),
                dtype='float64'
            )

            # 从projection层到输出层bias：[dictionary_size]
            proj_output_bias = tf.get_variable(
                'proj_output_bias',
                shape=[dictionary_size],
                initializer=tf.random_normal_initializer(stddev=cfg.stddev),
                dtype='float64'
            )

            self.output_layer = tf.nn.bias_add(tf.matmul(proj_layer, proj_output_weight), proj_output_bias)
            self.output_layer = tf.nn.softmax(self.output_layer)    # [cfg.batch_size, dictionary size]

    def train(self):
        with tf.device('/gpu:0'):
            # 构造loss
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output_layer, labels=self.one_hot_label))
            opt = tf.train.AdamOptimizer()
            grads_and_vars = opt.compute_gradients(self.loss)
            grads_and_vars = [(g, v) for (g, v) in grads_and_vars if g is not None]
            train_op = opt.apply_gradients(grads_and_vars)

            avg_loss = 0
            train_set_iteration = train_set_size / cfg.batch_size
            iteration_25 = train_set_iteration / 4
            iteration_50 = train_set_iteration / 2
            iteration_75 = train_set_iteration / 4 * 3
            print ('Train total batch size is %d, 25 is %d, 50 is %d, 75 is %d' % (train_set_iteration, iteration_25, iteration_50, iteration_75))
            for iter in range(train_set_iteration):
                if iter == iteration_25:
                    print ('25% complete')
                if iter == iteration_50:
                    print ('50% complete')
                if iter == iteration_75:
                    print ('75% complete')
                train_set = np.zeros((cfg.batch_size, cfg.embedding_window))
                train_set_label = np.zeros((cfg.batch_size, dictionary_size))
                for sub_iter in range(iter * cfg.batch_size, (iter + 1) * cfg.batch_size):
                    index = sub_iter - iter * cfg.batch_size
                    train_set_label[index][int(samples[sub_iter][sub_embedding_window])] = 1
                    train_set[index] = samples[sub_iter][0:sub_embedding_window] + samples[sub_iter][sub_embedding_window+1:cfg.embedding_window+1]
                if iter > 0:
                    tf.get_variable_scope().reuse_variables()
                if iter == train_set_size / cfg.batch_size - 1:
                    iter_loss, embed_weight = self.sess.run([self.loss, self.embed_weight],
                      feed_dict={self.input_samples: train_set,
                             self.one_hot_label: train_set_label})
                    output_embed_file = open(self.embed_file, 'w')
                    for embed_item in embed_weight:
                        embed_list = list(embed_item)
                        embed_list = [str(item) for item in embed_list]
                        output_embed_file.write(','.join(embed_list) + '\n')
                    output_embed_file.close()
                else:
                    iter_loss = self.sess.run(self.loss,
                        feed_dict={self.input_samples:train_set,
                                   self.one_hot_label:train_set_label})
                avg_loss += iter_loss
            print ('loss is %f' % (avg_loss / iter))

    def test(self):
        with tf.device('/gpu:0'):
            predict_result = tf.argmax(self.output_layer, dimension=1)    # [cfg.batch_size]
            predict_result = tf.cast(predict_result, dtype=tf.int32)
            comparision = tf.equal(predict_result, self.label)    # [cfg.batch_size]
            self.accuracy = tf.reduce_sum(tf.cast(comparision, dtype=tf.float32))

            total_accuracy = 0.0
            total_batch_size = sample_size / cfg.batch_size
            validate_set_iteration = train_set_size / cfg.batch_size + 1
            validate_batch_size = total_batch_size - train_set_size / cfg.batch_size
            iteration_25 = validate_batch_size / 4
            iteration_50 = validate_batch_size / 2
            iteration_75 = validate_batch_size / 4 * 3
            print ('Validation total batch size is %d, 25 is %d, 50 is %d, 75 is %d' % (validate_batch_size, iteration_25, iteration_50, iteration_75))
            for iter in range(validate_set_iteration, total_batch_size):
                if iter - validate_set_iteration == iteration_25:
                    print ('25% complete')
                if iter - validate_set_iteration == iteration_50:
                    print ('50% complete')
                if iter - validate_set_iteration == iteration_75:
                    print ('75% complete')
                validate_set = np.zeros((cfg.batch_size, cfg.embedding_window))
                validate_set_label = np.zeros((cfg.batch_size))
                for sub_iter in range(iter * cfg.batch_size, (iter + 1) * cfg.batch_size):
                    index = sub_iter - iter * cfg.batch_size
                    validate_set_label[index] = samples[sub_iter][sub_embedding_window]
                    validate_set[index] = samples[sub_iter][sub_embedding_window]
                accuracy = self.sess.run(self.accuracy, feed_dict={self.input_samples: validate_set, self.label: validate_set_label})
                total_accuracy += accuracy
            if validate_batch_size > 0:
                print('accuracy is %f' % (total_accuracy / validate_batch_size))

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "cbow <dictionary file> <onehot file> <embed file>"
        sys.exit()

    # 上(下)文窗口大小：sub_embedding_window
    sub_embedding_window = cfg.embedding_window / 2

    # placeholder不能嵌套使用
    # words_in_sentence = tf.placeholder(shape=(), dtype=tf.int32)
    # dictionary_size = tf.placeholder(shape=(), dtype=tf.int32)
    dict_file = open(sys.argv[1], 'rb').readlines()
    word_dict = {}
    reverse_word_dict = {}
    for line in dict_file:
        word_dict[line.split('\t')[0]] = line.split('\t')[1]
        reverse_word_dict[int(line.split('\t')[1])] = line.split('\t')[0]
    dictionary_size = len(word_dict) + 1

    sample_file = open(sys.argv[2], 'rb').readlines()
    samples = []
    sample_max_length = -1
    for line in sample_file:
        samples.append(line.replace('\n', '').split(','))
    sample_size = len(samples)
    # 根据batch_size截断样本
    sample_size = sample_size / cfg.batch_size * cfg.batch_size
    samples = samples[0:sample_size]
    train_set_size = int(sample_size / cfg.batch_size * cfg.train_set_ratio) * cfg.batch_size

    config = tf.ConfigProto(allow_soft_placement = True)
    with tf.Session(config=config) as sess:
        cbow_obj = CBOW(sess, sys.argv[3])
        sess.run(tf.global_variables_initializer())
        cbow_obj.train()
        cbow_obj.test()
        sess.close()