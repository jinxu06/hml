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
            # outputs = tf.layers.batch_normalization(outputs, training=is_training)
            outputs = tf.contrib.layers.batch_norm(outputs, is_training=is_training, reuse=True, scope=name+"-BN")
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
            W = tf.get_variable('W', shape=filter_size+[int(inputs.get_shape()[-1]), num_filters], dtype=tf.float32, trainable=True, initializer=kernel_initializer, regularizer=kernel_regularizer)
        if b is None:
            b = tf.get_variable('b', shape=[num_filters], dtype=tf.float32, trainable=True, initializer=tf.constant_initializer(0.), regularizer=None)

        outputs = tf.nn.bias_add(tf.nn.conv2d(inputs, W, [1] + stride + [1], pad), b)

        if bn:
            # outputs = tf.layers.batch_normalization(outputs, training=is_training)
            outputs = tf.contrib.layers.batch_norm(outputs, is_training=is_training, reuse=True, scope=name+"-BN")
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
            W = tf.get_variable('W', shape=filter_size+[num_filters, int(inputs.get_shape()[-1])], dtype=tf.float32, trainable=True, initializer=kernel_initializer, regularizer=kernel_regularizer)
        if b is None:
            b = tf.get_variable('b', shape=[num_filters], dtype=tf.float32, trainable=True, initializer=tf.constant_initializer(0.), regularizer=None)

        # calculate convolutional layer output
        outputs = tf.nn.conv2d_transpose(inputs, W, target_shape, [1] + stride + [1], padding=pad)
        outputs = tf.nn.bias_add(outputs, b)

        if bn:
            # outputs = tf.layers.batch_normalization(outputs, training=is_training)
            outputs = tf.contrib.layers.batch_norm(outputs, is_training=is_training, reuse=True, scope=name+"-BN")
        if nonlinearity is not None:
            outputs = nonlinearity(outputs)
        print("    + deconv2d", int_shape(inputs), int_shape(outputs), nonlinearity, bn)
        return outputs




"The implementation of PixelCNN here referes to PixelCNN++ from OpenAI"

def down_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0],1,xs[2],xs[3]]), x[:,:xs[1]-1,:,:]],1)

def right_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0],xs[1],1,xs[3]]), x[:,:,:xs[2]-1,:]],2)

def up_shift(x):
    xs = int_shape(x)
    return tf.concat([x[:,1:xs[1],:,:], tf.zeros([xs[0],1,xs[2],xs[3]])],1)

def left_shift(x):
    xs = int_shape(x)
    return tf.concat([x[:,:,1:xs[2],:], tf.zeros([xs[0],xs[1],1,xs[3]])],2)

@add_arg_scope
def down_shifted_conv2d(x, num_filters, W=None, b=None, filter_size=[2,3], strides=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[filter_size[0]-1,0], [int((filter_size[1]-1)/2),int((filter_size[1]-1)/2)],[0,0]])
    return conv2d(x, num_filters, W=W, b=b, kernel_size=filter_size, strides=strides, padding='VALID', **kwargs)

@add_arg_scope
def down_shifted_deconv2d(x, num_filters, W=None, b=None, filter_size=[2,3], strides=[1,1], **kwargs):
    x = deconv2d(x, num_filters, kernel_size=filter_size, strides=strides, padding='VALID', **kwargs)
    xs = int_shape(x)
    r = x[:,:(xs[1]-filter_size[0]+1),int((filter_size[1]-1)/2):(xs[2]-int((filter_size[1]-1)/2)),:]
    return r

@add_arg_scope
def down_right_shifted_conv2d(x, num_filters, W=None, b=None, filter_size=[2,2], strides=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[filter_size[0]-1, 0], [filter_size[1]-1, 0],[0,0]])
    return conv2d(x, num_filters, W=W, b=b, kernel_size=filter_size, strides=strides, padding='VALID', **kwargs)

@add_arg_scope
def down_right_shifted_deconv2d(x, num_filters, W=None, b=None, filter_size=[2,2], strides=[1,1], **kwargs):
    x = deconv2d(x, num_filters, W=W, b=b, kernel_size=filter_size, strides=strides, padding='VALID', **kwargs)
    xs = int_shape(x)
    r = x[:,:(xs[1]-filter_size[0]+1):,:(xs[2]-filter_size[1]+1),:]
    return r


@add_arg_scope
def up_shifted_conv2d(x, num_filters, W=None, b=None, filter_size=[2,3], strides=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[0, filter_size[0]-1], [int((filter_size[1]-1)/2),int((filter_size[1]-1)/2)],[0,0]])
    return conv2d(x, num_filters, W=W, b=b, kernel_size=filter_size, strides=strides, padding='VALID', **kwargs)

@add_arg_scope
def up_left_shifted_conv2d(x, num_filters, W=None, b=None, filter_size=[2,2], strides=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[0, filter_size[0]-1], [0, filter_size[1]-1],[0,0]])
    return conv2d(x, num_filters, W=W, b=b, kernel_size=filter_size, strides=strides, padding='VALID', **kwargs)


@add_arg_scope
def nin(x, num_units, W=None, b=None, **kwargs):
    """ a network in network layer (1x1 CONV) """
    s = int_shape(x)
    x = tf.reshape(x, [np.prod(s[:-1]),s[-1]])
    x = dense(x, num_units, W=W, b=b, **kwargs)
    return tf.reshape(x, s[:-1]+[num_units])

@add_arg_scope
def gated_resnet(x, a=None, gh=None, sh=None, params=None, nonlinearity=tf.nn.elu, conv=conv2d, dropout_p=0.0, counters={}, **kwargs):
    name = get_name("gated_resnet", counters)
    print("construct", name, "...")
    xs = int_shape(x)
    num_filters = xs[-1]
    with arg_scope([conv], **kwargs):
        if params is None:
            c1 = conv(nonlinearity(x), num_filters)
            if a is not None: # add short-cut connection if auxiliary input 'a' is given
                c1 += nin(nonlinearity(a), num_filters)
            c1 = nonlinearity(c1)
            c1 = tf.nn.dropout(c1, keep_prob=1. - dropout_p)
            c2 = conv(c1, num_filters * 2)
            # add projection of h vector if included: conditional generation
            if sh is not None:
                c2 += nin(sh, 2*num_filters, nonlinearity=nonlinearity)
            if gh is not None: # haven't finished this part
                pass
            a, b = tf.split(c2, 2, 3)
            c3 = a * tf.nn.sigmoid(b)
            return x + c3
        else:
            c1 = conv(nonlinearity(x), num_filters, W=params.pop(), b=params.pop())
            if a is not None: # add short-cut connection if auxiliary input 'a' is given
                c1 += nin(nonlinearity(a), num_filters, W=params.pop(), b=params.pop())
            c1 = nonlinearity(c1)
            c1 = tf.nn.dropout(c1, keep_prob=1. - dropout_p)
            c2 = conv(c1, num_filters * 2, W=params.pop(), b=params.pop())
            # add projection of h vector if included: conditional generation
            if sh is not None:
                c2 += nin(sh, 2*num_filters, W=params.pop(), b=params.pop(), nonlinearity=nonlinearity)
            if gh is not None: # haven't finished this part
                pass
            a, b = tf.split(c2, 2, 3)
            c3 = a * tf.nn.sigmoid(b)
            return x + c3
