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



class LangevinDynamicsVIProcess(object):

    def __init__(self, counters={}, user_mode='train'):
        self.counters = counters
        self.user_mode = user_mode

    def construct(self, sample_encoder, aggregator, conditional_decoder, task_type, obs_shape, r_dim, z_dim, label_shape=[], num_classes=1, alpha=0.01, inner_iters=1, eval_iters=5, nonlinearity=tf.nn.relu, bn=False, kernel_initializer=None, kernel_regularizer=None):
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
        self.alpha = alpha
        self.inner_iters = inner_iters
        self.eval_iters = eval_iters
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

        self._model(y_sigma=.2)

        self.loss = self._loss()
        self.grads = tf.gradients(self.loss, tf.trainable_variables(), colocate_gradients_with_ops=True)


    def _model(self, y_sigma=1.):
        default_args = {
            "nonlinearity": self.nonlinearity,
            "bn": self.bn,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "is_training": self.is_training,
            "counters": self.counters,
        }
        with arg_scope([self.conditional_decoder, self.sample_encoder, self.aggregator], **default_args):
            self.scope_name = get_name("mcmc_implicit_process", self.counters)
            with tf.variable_scope(self.scope_name):
                r_c = self.sample_encoder(self.X_c, self.y_c, self.r_dim, self.num_classes, bn=False)
                # z = self.aggregator(r_c, self.z_dim, bn=False)
                self.z_mu_pos, self.z_log_sigma_sq_pos = self.aggregator(r_c, self.z_dim, bn=False)
                z = gaussian_sampler(self.z_mu_pos, tf.exp(0.5*self.z_log_sigma_sq_pos))
                # z = tf.get_variable('z', shape=[1,self.z_dim], dtype=tf.float32, trainable=True, initializer=self.kernel_initializer, regularizer=self.kernel_regularizer)
                outputs  = self.conditional_decoder(self.X_c, z, reuse=False, counters={})
                self.outputs_sqs = [self.conditional_decoder(self.X_t, z, reuse=True, counters={})]
                loss_func = lambda z, o, y, beta: - (tf.reduce_sum(tf.distributions.Normal(loc=0., scale=y_sigma).log_prob(y-o)) \
                 + beta*tf.reduce_sum(tf.distributions.Normal(loc=0., scale=1.).log_prob(z)))
                for k in range(1, max(self.inner_iters, self.eval_iters)+1):
                    loss = loss_func(z, outputs, self.y_c, 1.0)
                    grad_z = tf.gradients(loss, z, colocate_gradients_with_ops=True)[0]
                    eta = tf.distributions.Normal(loc=0., scale=2*self.alpha).sample(sample_shape=int_shape(z))
                    z -= self.alpha * grad_z + eta
                    outputs = self.conditional_decoder(self.X_c, z, reuse=True, counters={})
                    outputs_t = self.conditional_decoder(self.X_t, z, reuse=True, counters={})
                    self.outputs_sqs.append(outputs_t)
                self.y_hat_sqs = [self.pred_func(o) for o in self.outputs_sqs]
                self.loss_sqs = [loss_func(z, o, self.y_t, 0.0) for o in self.outputs_sqs]
                self.mse_sqs = [self.error_func(self.y_t, o) for o in self.outputs_sqs]

    def _loss(self):
        return self.loss_sqs[self.inner_iters]

    def predict(self, X_c_value, y_c_value, X_t_value, step=None):
        feed_dict = {
            self.X_c: X_c_value,
            self.y_c: y_c_value,
            self.X_t: X_t_value,
            self.is_training: False,
        }
        if step is None:
            step = self.eval_iters
        return [self.y_hat_sqs[step]], feed_dict

    def evaluate_metrics(self, X_c_value, y_c_value, X_t_value, y_t_value, step=None):
        feed_dict = {
            self.X_c: X_c_value,
            self.y_c: y_c_value,
            self.X_t: X_t_value,
            self.y_t: y_t_value,
            self.is_training: False,
        }
        if step is None:
            step = self.eval_iters
        return [self.loss_sqs[step], self.mse_sqs[step]], feed_dict


@add_arg_scope
def fc_encoder(X, y, r_dim, num_classes=1, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    inputs = tf.concat([X, y[:, None]], axis=1)
    name = get_name("fc_encoder", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
            size = 256
            outputs = dense(inputs, size)
            outputs = nonlinearity(dense(outputs, size, nonlinearity=None) + dense(inputs, size, nonlinearity=None))
            inputs = outputs
            outputs = dense(outputs, size)
            outputs = nonlinearity(dense(outputs, size, nonlinearity=None) + dense(inputs, size, nonlinearity=None))
            outputs = dense(outputs, size)
            outputs = dense(outputs, r_dim, nonlinearity=None, bn=False)
            return outputs

@add_arg_scope
def aggregator(r, z_dim, method=tf.reduce_mean, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("aggregator", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
            r = method(r, axis=0, keepdims=True)
            size = 128
            r = dense(r, size)
            r = dense(r, size)
            # z = dense(r, z_dim, nonlinearity=None, bn=False)
            # return z
            z_mu = dense(r, z_dim, nonlinearity=None, bn=False)
            z_log_sigma_sq = dense(r, z_dim, nonlinearity=None, bn=False)
            return z_mu, z_log_sigma_sq



@add_arg_scope
def conditional_decoder(x, z, num_classes=1, reuse=False, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("conditional_decoder", counters)
    print("construct", name, "...")
    with tf.variable_scope(name, reuse=reuse):
        with arg_scope([dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
            size = 256
            batch_size = tf.shape(x)[0]
            x = tf.tile(x, tf.stack([1, int_shape(z)[1]]))
            z = tf.tile(z, tf.stack([batch_size, 1]))
            # xz = x + z * tf.get_variable(name="coeff", shape=(), dtype=tf.float32, initializer=tf.constant_initializer(2.0))
            xz = x
            a = dense(xz, size, nonlinearity=None) + dense(z, size, nonlinearity=None)
            outputs = tf.nn.tanh(a) * tf.sigmoid(a)

            for k in range(4):
                a = dense(outputs, size, nonlinearity=None) + dense(z, size, nonlinearity=None)
                outputs = tf.nn.tanh(a) * tf.sigmoid(a)
            outputs = dense(outputs, 1, nonlinearity=None, bn=False)
            outputs = tf.reshape(outputs, shape=(batch_size,))
            return outputs
