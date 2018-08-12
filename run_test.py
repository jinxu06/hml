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

from misc.estimators import estimate_kld, compute_gaussian_kld


# initializer = tf.global_variables_initializer()
# saver = tf.train.Saver()

bsize_x = 100
bsize_y = 50
d = 32

x_ph = tf.placeholder(dtype=tf.float32, shape=[bsize_x, d])
y_ph = tf.placeholder(dtype=tf.float32, shape=[bsize_y, d])
kld = estimate_kld(x_ph, y_ph)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    if args.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(initializer)


    x = np.random.normal(0.0, 1.0, size=(bsize_x, d))
    y = np.random.normal(0.0, 1.0, size=(bsize_y, d))

    feed_dict = {
        x_ph: x,
        y_ph: y
    }
    print(sess.run(kld, feed_dict=feed_dict))
