"""
Various tensorflow utilities
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from misc.helpers import int_shape, get_name


@add_arg_scope
def dense(inputs, num_outputs, W=None, b=None, nonlinearity=None, bn=False, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    ''' fully connected layer '''
    name = get_name('dense', counters)
    with tf.variable_scope(name):
        if W is None:
            W = tf.get_variable('W', shape=[int(inputs.get_shape()[1]),num_outputs], dtype=tf.float32, trainable=True, initializer=kernel_initializer, regularizer=kernel_regularizer)
        if b is None:
            b = tf.get_variable('b', shape=[num_outputs], dtype=tf.float32, trainable=True, initializer=tf.constant_initializer(0.), regularizer=None)

        outputs = tf.matmul(inputs, W) + tf.reshape(b, [1, num_outputs])

        if bn:
            outputs = tf.layers.batch_normalization(outputs, training=is_training)
        if nonlinearity is not None:
            outputs = nonlinearity(outputs)
        print("    + dense", int_shape(inputs), int_shape(outputs), nonlinearity, bn)
        return outputs

@add_arg_scope
def conv2d(inputs, num_filters, W=None, b=None, filter_size=[3,3], stride=[1,1], pad='SAME', nonlinearity=None, bn=False, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    ''' convolutional layer '''
    name = get_name('conv2d', counters)
    with tf.variable_scope(name):

        if W is None:
            W = tf.get_variable('W', shape=filter_size+[int(x.get_shape()[-1]), num_filters], dtype=tf.float32, trainable=True, initializer=kernel_initializer, regularizer=kernel_regularizer)
        if b is None:
            b = tf.get_variable('b', shape=[num_filters], dtype=tf.float32, trainable=True, initializer=tf.constant_initializer(0.), regularizer=None)

        outputs = tf.nn.bias_add(tf.nn.conv2d(inputs, W, [1] + stride + [1], pad), b)

        if bn:
            outputs = tf.layers.batch_normalization(outputs, training=is_training)
        if nonlinearity is not None:
            outputs = nonlinearity(outputs)
        print("    + conv2d", int_shape(inputs), int_shape(outputs), nonlinearity, bn)
        return outputs


@add_arg_scope
def deconv2d(inputs, num_filters, W=None, b=None, filter_size=[3,3], stride=[1,1], pad='SAME', nonlinearity=None, bn=False, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    ''' transposed convolutional layer '''
    name = get_name('deconv2d', counters)
    xs = int_shape(inputs)
    if pad=='SAME':
        target_shape = [xs[0], xs[1]*stride[0], xs[2]*stride[1], num_filters]
    else:
        target_shape = [xs[0], xs[1]*stride[0] + filter_size[0]-1, xs[2]*stride[1] + filter_size[1]-1, num_filters]
    with tf.variable_scope(name):
        if W is None:
            W = tf.get_variable('W', shape=filter_size+[num_filters, int(x.get_shape()[-1])], dtype=tf.float32, trainable=True, initializer=kernel_initializer, regularizer=kernel_regularizer)
        if b is None:
            b = tf.get_variable('b', shape=[num_filters], dtype=tf.float32, trainable=True, initializer=tf.constant_initializer(0.), regularizer=None)

        # calculate convolutional layer output
        outputs = tf.nn.conv2d_transpose(inputs, W, target_shape, [1] + stride + [1], padding=pad)
        outputs = tf.nn.bias_add(outputs, b)

        if bn:
            outputs = tf.layers.batch_normalization(outputs, training=is_training)
        if nonlinearity is not None:
            outputs = nonlinearity(outputs)
        print("    + deconv2d", int_shape(inputs), int_shape(outputs), nonlinearity, bn)
        return outputs
