import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_integer('embedding_size', 256, 'embedding size')
flags.DEFINE_integer('embedding_window', 4, 'embedding window')
flags.DEFINE_integer('hidden_size', 128, 'word2vec weight size')
flags.DEFINE_float('train_set_ratio', 1.0, 'train set ratio')
flags.DEFINE_integer('batch_size', 256, 'train batch size')
flags.DEFINE_integer('top_k_sim', -5, 'top k similarity items')

flags.DEFINE_float('stddev', 0.01, 'stddev for W initializer')
flags.DEFINE_integer('negative_sample_size', 5, 'negative sample size')
flags.DEFINE_integer('MAX_GRAD_NORM', 5, 'maximum gradient norm')

cfg = tf.app.flags.FLAGS
