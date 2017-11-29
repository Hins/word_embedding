# -*-coding:utf-8-*-
import numpy as np
import tensorflow as tf
from config import cfg

'''
分组训练word2vec：将一句话拆分成cfg.words_in_sentence - cfg.embedding_window个子样本，每个子样本长度为cfg.embedding_window + 1
潘晓彤
2017.11.24
'''

# 上(下)文窗口大小：sub_embedding_window
sub_embedding_window = cfg.embedding_window / 2
train_set = np.random.randint(0, cfg.dictionary_size - 1, size=[cfg.train_batch_size, cfg.words_in_sentence])    # 100 words in one sentence
train_set_label = train_set[:,sub_embedding_window:cfg.words_in_sentence - sub_embedding_window]
np_zeros = np.zeros((cfg.train_batch_size, cfg.words_in_sentence - cfg.embedding_window, cfg.dictionary_size))
for b in range(cfg.dictionary_size):
    for i in range(cfg.words_in_sentence - cfg.embedding_window):
        np_zeros[b][i][train_set[b][i]] = 1
train_set_label = np_zeros

# with tf.variable_scope("dialogue", reuse=True):
samples = tf.placeholder(shape=[cfg.words_in_sentence], dtype=tf.int32)
label = tf.placeholder(shape=[cfg.words_in_sentence - cfg.embedding_window, cfg.dictionary_size], dtype=tf.int32)

# 特征矩阵：V * m
# [num_class, embedding_dimension]
embed_weight = tf.get_variable(
    'embed_weight',
    shape=(cfg.dictionary_size, cfg.embedding_size),
    initializer=tf.random_normal_initializer(stddev=cfg.stddev),
    dtype='float64'
)

# 特征矩阵bias：V
embed_bias = tf.get_variable(
    'embed_bias',
    shape=(cfg.dictionary_size),
    initializer=tf.random_normal_initializer(stddev=cfg.stddev),
    dtype='float64'
)

# [256, 100, 256]: batch_size=256, words_in_sentence=100, embedding_dimension=256
embed_init = tf.nn.embedding_lookup(embed_weight, samples)

# 从embedding到hidden层权重矩阵：[h, (n-1)*m]
# [128,1024]
proj_hidden_weight = tf.get_variable(
    'proj_hidden_weight',
    shape=(cfg.hidden_size, cfg.embedding_window * cfg.embedding_size),
    initializer=tf.random_normal_initializer(stddev=cfg.stddev),
    dtype='float64'
)

# 从embedding到hidden层bias：[h]
# [128]
proj_hidden_bias = tf.get_variable(
    'proj_hidden_bias',
    shape=(cfg.hidden_size),
    initializer=tf.random_normal_initializer(stddev=cfg.stddev),
    dtype='float64'
)

# 从hidden层到输出层权重矩阵：[V, h]
# [10, 128]
hidden_output_weight = tf.get_variable(
    'hidden_output_weight',
    shape=(cfg.dictionary_size, cfg.hidden_size),
    initializer=tf.random_normal_initializer(stddev=cfg.stddev),
    dtype='float64'
)

# 从hidden层到输出层bias：[V]
# [10]
hidden_output_bias = tf.get_variable(
    'hidden_output_bias',
    shape=(cfg.dictionary_size),
    initializer=tf.random_normal_initializer(stddev=cfg.stddev),
    dtype='float64'
)

# 将一个样本中的embedding matrix平摊成embedding vector
proj_layer = tf.concat(axis=0, values=[
    tf.reshape(embed_init, shape=[-1])
])
sample_list = []
sample_size = cfg.words_in_sentence - cfg.embedding_window
for i in range(sample_size):
    sample_list.append(tf.concat(
        axis=0,
        values=[proj_layer[i*cfg.embedding_size : (i+sub_embedding_window)*cfg.embedding_size],
                proj_layer[(i + sub_embedding_window + 1) * cfg.embedding_size: (i + cfg.embedding_window + 1) * cfg.embedding_size]]
    ))
# [cfg.batch_size, cfg.embedding_window * cfg.embedding_size, cfg.words_in_sentence - cfg.embedding_window]: [256, 1024, 96]
proj_layer = tf.reshape(sample_list, shape=[-1, sample_size])
proj_layer = tf.matmul(proj_hidden_weight, tf.nn.tanh(proj_layer))
proj_layer = tf.reshape(proj_layer, shape=[sample_size, -1])
proj_layer = tf.nn.bias_add(proj_layer, proj_hidden_bias)
proj_layer = tf.reshape(proj_layer, shape=[cfg.hidden_size, -1])
hidden_layer = tf.matmul(hidden_output_weight, proj_layer)
hidden_layer = tf.reshape(hidden_layer, shape=[-1, cfg.dictionary_size])
hidden_layer = tf.nn.bias_add(hidden_layer, hidden_output_bias)
output_layer = tf.nn.softmax(hidden_layer)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=label))
predict_result = tf.argmax(output_layer, dimension=1)

opt = tf.train.AdamOptimizer()
grads_and_vars = opt.compute_gradients(loss)
grads_and_vars = [(g, v) for (g, v) in grads_and_vars if g is not None]
train_op = opt.apply_gradients(grads_and_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # train
    avg_loss = 0
    for iter in range(cfg.train_batch_size):
        iter_loss = sess.run(loss, feed_dict={samples: train_set[iter], label: train_set_label[iter]})
        avg_loss += iter_loss
    print ('loss is %f', avg_loss / iter)

    # validation
    validate_set = np.random.randint(0, cfg.dictionary_size - 1, size=[cfg.test_batch_size, cfg.words_in_sentence])
    validate_set_label = validate_set[:, sub_embedding_window:cfg.words_in_sentence - sub_embedding_window]
    np_zeros = np.zeros((cfg.test_batch_size, cfg.words_in_sentence - cfg.embedding_window, cfg.dictionary_size))
    for b in range(cfg.dictionary_size):
        for i in range(cfg.words_in_sentence - cfg.embedding_window):
            np_zeros[b][i][validate_set[b][i]] = 1
    validate_set_one_hot_label = np_zeros
    total_accuracy = 0
    for iter in range(cfg.test_batch_size):
        comparision = tf.equal(predict_result, validate_set_label[iter])
        accuracy = tf.reduce_mean(tf.cast(comparision, dtype=tf.float32))
        accuracy = sess.run(accuracy, feed_dict={samples: validate_set[iter], label: validate_set_one_hot_label[iter]})
        total_accuracy += accuracy
    print('accuracy is %.3f', accuracy / iter)
    sess.close()