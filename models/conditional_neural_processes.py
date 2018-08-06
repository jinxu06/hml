import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from misc.layers import conv2d, deconv2d, dense
from misc.samplers import gaussian_sampler
from misc.helpers import int_shape, get_name, get_trainable_variables
from misc.estimators import compute_2gaussian_kld
from misc.losses import mean_squared_error
from misc.metrics import accuracy



class ConditionalNeuralProcess(object):

    def __init__(self, counters={}, user_mode='train'):
        self.counters = counters
        self.user_mode = user_mode

    def construct(self, sample_encoder, aggregator, conditional_decoder, task_type, obs_shape, r_dim, z_dim, label_shape=[], num_classes=1, nonlinearity=tf.nn.relu, bn=False, kernel_initializer=None, kernel_regularizer=None):
        #
        self.sample_encoder = sample_encoder
        self.aggregator = aggregator
        self.conditional_decoder = conditional_decoder
        self.task_type = task_type
        if task_type == 'classification':
            self.error_func = tf.losses.softmax_cross_entropy
            self.pred_func = lambda x: tf.nn.softmax(x)
        elif task_type == 'regression':
            self.error_func = tf.losses.mean_squared_error
            self.pred_func = lambda x: x
        else:
            raise Exception("Unknown task type")
        self.obs_shape = obs_shape
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.label_shape = label_shape
        self.num_classes = num_classes
        self.nonlinearity = nonlinearity
        self.bn = bn
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        #
        self.X_c = tf.placeholder(tf.float32, shape=tuple([None,]+obs_shape))
        self.y_c = tf.placeholder(tf.float32, shape=tuple([None,]+label_shape))
        self.X_t = tf.placeholder(tf.float32, shape=tuple([None,]+obs_shape))
        self.y_t = tf.placeholder(tf.float32, shape=tuple([None,]+label_shape))
        self.is_training = tf.placeholder(tf.bool, shape=())

        self._model()
        self.loss = self._loss(beta=1.0, y_sigma=0.2)
        self.grads = tf.gradients(self.loss, tf.trainable_variables(), colocate_gradients_with_ops=True)

        #
        # self.outputs = self._model()
        # self.y_hat = self.outputs
        # self.loss = self._loss(beta=1.0, y_sigma=0.2)
        # #
        # self.grads = tf.gradients(self.loss, get_trainable_variables([self.scope_name]), colocate_gradients_with_ops=True)


    def _model(self):
        default_args = {
            "nonlinearity": self.nonlinearity,
            "bn": self.bn,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "is_training": self.is_training,
            "counters": self.counters,
        }
        with arg_scope([self.conditional_decoder], **default_args):
            default_args.update({"bn":False})
            with arg_scope([self.sample_encoder, self.aggregator], **default_args):
                self.scope_name = get_name("neural_process", self.counters)
                with tf.variable_scope(self.scope_name):
                    num_c = tf.shape(self.X_c)[0]
                    X_ct = tf.concat([self.X_c, self.X_t], axis=0)
                    y_ct = tf.concat([self.y_c, self.y_t], axis=0)
                    r_ct = self.sample_encoder(X_ct, y_ct, self.r_dim, self.num_classes)
                    self.r_ct = r_ct
                    self.z_mu_pr, self.z_log_sigma_sq_pr, self.z_mu_pos, self.z_log_sigma_sq_pos = self.aggregator(r_ct, num_c, self.z_dim)
                    if self.user_mode == 'train':
                        z = self.z_mu_pos
                        ## z = gaussian_sampler(self.z_mu_pos, tf.exp(0.5*self.z_log_sigma_sq_pos))
                    elif self.user_mode == 'eval':
                        z = self.z_mu_pos
                    else:
                        raise Exception("unknown user_mode")
                    z = (1-self.use_z_ph) * z + self.use_z_ph * self.z_ph
                    self.outputs = self.conditional_decoder(self.X_t, z, self.num_classes)

    def _loss(self):
        self.nll = self.error_func(self.y_t, self.outputs)
        return self.nll

    def predict(self, X_c_value, y_c_value, X_t_value):
        feed_dict = {
            self.X_c: X_c_value,
            self.y_c: y_c_value,
            self.X_t: X_t_value,
            self.is_training: False,
        }
        preds = self.pred_func(self.outputs)
        return [preds], feed_dict

    def evaluate_metrics(self, X_c_value, y_c_value, X_t_value, y_t_value):
        feed_dict = {
            self.X_c: X_c_value,
            self.y_c: y_c_value,
            self.X_t: X_t_value,
            self.y_t: y_t_value,
            self.is_training: False,
        }
        if self.task_type == 'classification':
            return [self.loss, accuracy(self.y_t, self.pred_func(self.outputs))], feed_dict
        return [self.loss], feed_dict


@add_arg_scope
def omniglot_conv_encoder(X, y, r_dim, num_classes, is_training, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("omniglot_conv_encoder", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        default_args = {
            "nonlinearity": nonlinearity,
            "bn": bn,
            "kernel_initializer": kernel_initializer,
            "kernel_regularizer": kernel_regularizer,
            "is_training": is_training,
            "counters": counters,
        }
        num_filters = 64
        filter_size = [3, 3]
        stride = [2, 2]
        bsize = tf.shape(X)[0]
        with arg_scope([conv2d, dense], **default_args):
            outputs = X

            for _ in range(4):
                outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
            outputs = tf.reshape(outputs, [-1, np.prod(int_shape(outputs)[1:])])
            outputs = tf.concat([outputs, y], axis=-1)
            outputs = dense(outputs, num_filters)
            r = dense(outputs, r_dim, nonlinearity=None, bn=False)
            return r


@add_arg_scope
def omniglot_conv_conditional_decoder(inputs, z, num_classes, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("omniglot_conv_conditional_decoder", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        default_args = {
            "nonlinearity": nonlinearity,
            "bn": bn,
            "kernel_initializer": kernel_initializer,
            "kernel_regularizer": kernel_regularizer,
            "is_training": is_training,
            "counters": counters,
        }
        num_filters = 64
        filter_size = [3, 3]
        stride = [2, 2]
        bsize = tf.shape(inputs)[0]
        with arg_scope([conv2d, dense], **default_args):
            outputs = inputs

            for _ in range(4):
                outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
            outputs = tf.reshape(outputs, [-1, np.prod(int_shape(outputs)[1:])])
            z = tf.tile(z, tf.stack([bsize, 1]))
            outputs = tf.concat([outputs, z], axis=-1)
            outputs = dense(outputs, num_filters)
            y = dense(outputs, num_classes, nonlinearity=None, bn=False)
            return y


@add_arg_scope
def aggregator(r, num_c, z_dim, method=tf.reduce_mean, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("aggregator", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
            r_pr = method(r[:num_c], axis=0, keepdims=True)
            r = method(r, axis=0, keepdims=True)
            r = tf.concat([r_pr, r], axis=0)
            size = 256
            r = dense(r, size)
            r = dense(r, size)
            r = dense(r, size)
            z_mu = dense(r, z_dim, nonlinearity=None, bn=False)
            z_log_sigma_sq = dense(r, z_dim, nonlinearity=None, bn=False)
            return z_mu[:1], z_log_sigma_sq[:1], z_mu[1:], z_log_sigma_sq[1:]
