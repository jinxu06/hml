import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from misc.helpers import log_prob_from_logits, int_shape, get_name, log_sum_exp
flatten = tf.contrib.layers.flatten

def accuracy(y, preds):
    cmp = tf.equal(tf.argmax(y, 1), tf.argmax(preds, 1))
    cmp = tf.cast(cmp, tf.int32)
    return tf.reduce_mean(cmp)
