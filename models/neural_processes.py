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



class NeuralProcess(object):

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
            self.error_func = mean_squared_error #tf.losses.mean_squared_error
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
        self.use_z_pr = tf.cast(tf.placeholder_with_default(False, shape=()), dtype=tf.float32)

        self._model()
        self.loss = self._loss(beta=1.0, y_sigma=0.2) # y_sigma=0.2)
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
            with arg_scope([self.sample_encoder,self.aggregator], **default_args):
                self.scope_name = get_name("neural_process", self.counters)
                with tf.variable_scope(self.scope_name):
                    num_c = tf.shape(self.X_c)[0]
                    X_ct = tf.concat([self.X_c, self.X_t], axis=0)
                    y_ct = tf.concat([self.y_c, self.y_t], axis=0)
                    r_ct = self.sample_encoder(X_ct, y_ct, self.r_dim, self.num_classes)
                    self.r_ct = r_ct
                    if self.task_type == 'classification':
                        self.z_mu_pr, self.z_log_sigma_sq_pr, self.z_mu_pos, self.z_log_sigma_sq_pos = self.aggregator(r_ct, y_ct, num_c, self.z_dim)
                    else:
                        self.z_mu_pr, self.z_log_sigma_sq_pr, self.z_mu_pos, self.z_log_sigma_sq_pos = self.aggregator(r_ct, num_c, self.z_dim)
                    if self.user_mode == 'train':
                        # z = self.z_mu_pr ##
                        z = gaussian_sampler(self.z_mu_pos, tf.exp(0.5*self.z_log_sigma_sq_pos))
                        z_pr = gaussian_sampler(self.z_mu_pr, tf.exp(0.5*self.z_log_sigma_sq_pr))
                    elif self.user_mode == 'eval':
                        z = self.z_mu_pos
                    else:
                        raise Exception("unknown user_mode")
                    z = (1-self.use_z_pr) * z + self.use_z_pr * z_pr
                    self.outputs = self.conditional_decoder(self.X_t, z, self.num_classes)
                    self.mse = tf.losses.mean_squared_error(self.y_t, self.outputs)

    def _loss(self, beta=1., y_sigma=1./np.sqrt(2)):
        self.reg = compute_2gaussian_kld(self.z_mu_pr, self.z_log_sigma_sq_pr, self.z_mu_pos, self.z_log_sigma_sq_pos)
        # self.nll = mean_squared_error(self.y_t, self.y_hat, sigma=y_sigma)
        self.nll = self.error_func(self.y_t, self.outputs) / (2*y_sigma**2)
        return self.nll + beta * self.reg

    def predict(self, X_c_value, y_c_value, X_t_value):
        feed_dict = {
            self.X_c: X_c_value,
            self.y_c: y_c_value,
            self.X_t: X_t_value,
            self.y_t: np.zeros((X_t_value.shape[0],)),
            self.is_training: False,
            self.use_z_pr: True,
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
        return [self.loss, self.mse], feed_dict

@add_arg_scope
def miniimagenet_conv_encoder(X, y, r_dim, num_classes, is_training, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("miniimagenet_conv_encoder", counters)
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
        batch_size = tf.shape(X)[0]
        with arg_scope([conv2d, dense], **default_args):
            outputs = X
            for _ in range(4):
                outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
            outputs = tf.reshape(outputs, [-1, np.prod(int_shape(outputs)[1:])])
            # outputs = tf.concat([outputs, y], axis=-1)
            # outputs = dense(outputs, num_filters)
            r = dense(outputs, r_dim, nonlinearity=None, bn=False)
            return r

@add_arg_scope
def miniimagenet_conv_conditional_decoder(inputs, z, num_classes, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("miniimagenet_conv_conditional_decoder", counters)
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

            z_tile = tf.tile(tf.reshape(z, [1, 1, 1, int_shape(z)[-1]]), tf.stack([bsize, 84, 84, 1]))
            outputs = tf.concat([outputs, z_tile], axis=-1)
            outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
            #
            z_tile = tf.tile(tf.reshape(z, [1, 1, 1, int_shape(z)[-1]]), tf.stack([bsize, 42, 42, 1]))
            outputs = tf.concat([outputs, z_tile], axis=-1)
            outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
            #
            z_tile = tf.tile(tf.reshape(z, [1, 1, 1, int_shape(z)[-1]]), tf.stack([bsize, 21, 21, 1]))
            outputs = tf.concat([outputs, z_tile], axis=-1)
            outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
            #
            z_tile = tf.tile(tf.reshape(z, [1, 1, 1, int_shape(z)[-1]]), tf.stack([bsize, 11, 11, 1]))
            outputs = tf.concat([outputs, z_tile], axis=-1)
            outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
            #
            outputs = tf.reduce_mean(outputs, [1, 2])
            outputs = tf.reshape(outputs, [-1, num_filters])
            y = dense(outputs, num_classes, nonlinearity=None, bn=False)
            return y



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
        batch_size = tf.shape(X)[0]
        with arg_scope([conv2d, dense], **default_args):
            outputs = X
            for _ in range(4):
                outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
            outputs = tf.reshape(outputs, [-1, np.prod(int_shape(outputs)[1:])])
            # outputs = tf.concat([outputs, y], axis=-1)
            # outputs = dense(outputs, num_filters)
            r = dense(outputs, r_dim, nonlinearity=None, bn=False)
            return r



            # #
            # y_tile = tf.tile(tf.reshape(y, [-1, 1, 1, num_classes]), tf.stack([1, 28, 28, 1]))
            # outputs = tf.concat([outputs, y_tile], axis=-1)
            # outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
            # #
            # y_tile = tf.tile(tf.reshape(y, [-1, 1, 1, num_classes]), tf.stack([1, 14, 14, 1]))
            # outputs = tf.concat([outputs, y_tile], axis=-1)
            # outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
            # #
            # y_tile = tf.tile(tf.reshape(y, [-1, 1, 1, num_classes]), tf.stack([1, 7, 7, 1]))
            # outputs = tf.concat([outputs, y_tile], axis=-1)
            # outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
            # #
            # y_tile = tf.tile(tf.reshape(y, [-1, 1, 1, num_classes]), tf.stack([1, 4, 4, 1]))
            # outputs = tf.concat([outputs, y_tile], axis=-1)
            # outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
            # #
            # outputs = tf.reduce_mean(outputs, [1, 2])
            # outputs = tf.reshape(outputs, [-1, num_filters])
            # r = dense(outputs, r_dim, nonlinearity=None, bn=False)
            #
            # return r


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

            # for _ in range(4):
            #     outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
            # outputs = tf.reshape(outputs, [-1, np.prod(int_shape(outputs)[1:])])
            # z = tf.tile(z, tf.stack([bsize, 1]))
            # outputs = tf.concat([outputs, z], axis=-1)
            # outputs = dense(outputs, num_filters)
            # y = dense(outputs, num_classes, nonlinearity=None, bn=False)
            # return y


            z_tile = tf.tile(tf.reshape(z, [1, 1, 1, int_shape(z)[-1]]), tf.stack([bsize, 28, 28, 1]))
            outputs = tf.concat([outputs, z_tile], axis=-1)
            outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
            #
            z_tile = tf.tile(tf.reshape(z, [1, 1, 1, int_shape(z)[-1]]), tf.stack([bsize, 14, 14, 1]))
            outputs = tf.concat([outputs, z_tile], axis=-1)
            outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
            #
            z_tile = tf.tile(tf.reshape(z, [1, 1, 1, int_shape(z)[-1]]), tf.stack([bsize, 7, 7, 1]))
            outputs = tf.concat([outputs, z_tile], axis=-1)
            outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
            #
            z_tile = tf.tile(tf.reshape(z, [1, 1, 1, int_shape(z)[-1]]), tf.stack([bsize, 4, 4, 1]))
            outputs = tf.concat([outputs, z_tile], axis=-1)
            outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
            #
            outputs = tf.reduce_mean(outputs, [1, 2])
            outputs = tf.reshape(outputs, [-1, num_filters])
            y = dense(outputs, num_classes, nonlinearity=None, bn=False)
            return y


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



@add_arg_scope
def cls_aggregator(r, y, num_c, z_dim, method=tf.reduce_mean, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("cls_aggregator", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
            r_dim = int_shape(r)[-1]
            num_classes = int_shape(y)[-1]
            r_pr_arr, r_arr = [], []
            for k in range(num_classes):
                r_pr_arr.append(method(tf.tile(y[:num_c, k:k+1], [1,r_dim]) * r[:num_c], axis=0, keepdims=True))
                r_arr.append(method(tf.tile(y[:, k:k+1], [1,r_dim]) * r, axis=0, keepdims=True))
            r_pr = tf.concat(r_pr_arr, axis=0)
            r = tf.concat(r_arr, axis=0)
            r = tf.concat([r_pr, r], axis=0)

            num_units = 256
            r = dense(r, num_units)
            r = dense(r, num_units)
            z_mu = dense(r, z_dim//num_classes, nonlinearity=None, bn=False)
            z_log_sigma_sq = dense(r, z_dim//num_classes, nonlinearity=None, bn=False)

            z_mu_pr, z_mu_pos = z_mu[:num_classes], z_mu[num_classes:]
            z_log_sigma_sq_pr, z_log_sigma_sq_pos = z_log_sigma_sq[:num_classes], z_log_sigma_sq[num_classes:]

            f = lambda x: tf.expand_dims(tf.concat(tf.unstack(x, axis=0), axis=0), axis=0)
            z_mu_pr = f(z_mu_pr)
            z_mu_pos = f(z_mu_pos)
            z_log_sigma_sq_pr = f(z_log_sigma_sq_pr)
            z_log_sigma_sq_pos = f(z_log_sigma_sq_pos)
            return z_mu_pr, z_log_sigma_sq_pr, z_mu_pos, z_log_sigma_sq_pos


            # r_pr_arr, r_arr = [], []
            # for k in range(int_shape(y)[-1]):
            #     r_pr_arr.append(method(tf.tile(y[:num_c, k:k+1], [1,r_dim]) * r[:num_c], axis=0, keepdims=True))
            #     r_arr.append(method(tf.tile(y[:, k:k+1], [1,r_dim]) * r, axis=0, keepdims=True))
            # r_pr = tf.concat(r_pr_arr, axis=-1)
            # r = tf.concat(r_arr, axis=-1)
            # r = tf.concat([r_pr, r], axis=0)
            # size = 256
            # r = dense(r, size)
            # r = dense(r, size)
            # r = dense(r, size)
            # z_mu = dense(r, z_dim, nonlinearity=None, bn=False)
            # z_log_sigma_sq = dense(r, z_dim, nonlinearity=None, bn=False)
            # # return z_mu[:1], z_log_sigma_sq[:1], z_mu[1:], z_log_sigma_sq[1:]
            # return r[:1], z_log_sigma_sq[:1], r[1:], z_log_sigma_sq[1:]


@add_arg_scope
def conditional_decoder(x, z, num_classes=1, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("conditional_decoder", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
            size = 256
            x_dim = int_shape(x)[-1]
            batch_size = tf.shape(x)[0]
            x = tf.tile(x, tf.stack([1, int_shape(z)[1]//x_dim]))
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




@add_arg_scope
def fc_encoder_2d(X, y, r_dim, num_classes=1, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    inputs = tf.concat([X, y[:, None]], axis=1)
    name = get_name("fc_encoder", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
            size = 512
            outputs = dense(inputs, size)
            outputs = nonlinearity(dense(outputs, size, nonlinearity=None) + dense(inputs, size, nonlinearity=None))
            inputs = outputs
            outputs = dense(outputs, size)
            outputs = nonlinearity(dense(outputs, size, nonlinearity=None) + dense(inputs, size, nonlinearity=None))
            outputs = dense(outputs, size)
            outputs = dense(outputs, r_dim, nonlinearity=None, bn=False)
            return outputs

@add_arg_scope
def aggregator_2d(r, num_c, z_dim, method=tf.reduce_mean, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("aggregator", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
            r_pr = method(r[:num_c], axis=0, keepdims=True)
            r = method(r, axis=0, keepdims=True)
            r = tf.concat([r_pr, r], axis=0)
            size = 512
            r = dense(r, size)
            r = dense(r, size)
            r = dense(r, size)
            z_mu = dense(r, z_dim, nonlinearity=None, bn=False)
            z_log_sigma_sq = dense(r, z_dim, nonlinearity=None, bn=False)
            return z_mu[:1], z_log_sigma_sq[:1], z_mu[1:], z_log_sigma_sq[1:]

@add_arg_scope
def conditional_decoder_2d(x, z, num_classes=1, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("conditional_decoder", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
            size = 512
            x_dim = int_shape(x)[-1]
            batch_size = tf.shape(x)[0]
            x = tf.tile(x, tf.stack([1, int_shape(z)[1]//x_dim]))
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
