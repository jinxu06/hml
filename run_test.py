import matplotlib
matplotlib.use('Agg')
import os
import sys
import json
import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

y_sigma = 0.01
error_func = tf.losses.mean_squared_error
loss_func1 = lambda o, y: error_func(y, o) / (2*y_sigma**2)
loss_func2 = lambda o, y: - tf.reduce_sum(tf.distributions.Normal(loc=0., scale=y_sigma).log_prob(y-o))

targets = tf.placeholder(tf.float32, shape=[None, 1])
preds = tf.placeholder(tf.float32, shape=[None, 1])

l1 = loss_func1(preds, targets)
l2 = loss_func2(preds, targets)


initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    if args.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(initializer)

    feed_dict = {
        targets: np.random.normal(size=(32, 1))
        preds: np.random.normal(size=(32, 1))
    }
    print(sess.run(l1, feed_dict=feed_dict))
    print(sess.run(l2, feed_dict=feed_dict))
