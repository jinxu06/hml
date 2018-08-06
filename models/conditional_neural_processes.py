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
        self.loss = self._loss()
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
            #default_args.update({"bn":False})
            with arg_scope([self.sample_encoder, self.aggregator], **default_args):
                self.scope_name = get_name("neural_process", self.counters)
                with tf.variable_scope(self.scope_name):
                    r_c = self.sample_encoder(self.X_c, self.y_c, self.r_dim, self.num_classes)
                    self.r_c = r_c
                    if self.task_type == 'classification':
                        self.r = self.aggregator(r_c, self.y_c, self.z_dim)
                    self.outputs = self.conditional_decoder(self.X_t, self.r, self.num_classes)
                    self.preds = self.pred_func(self.outputs)

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


    # def predict(self, sess, X_c_value, y_c_value, X_t_value):
    #     feed_dict = {
    #         self.X_c: X_c_value,
    #         self.y_c: y_c_value,
    #         self.X_t: X_t_value,
    #         self.y_t: np.zeros((X_t_value.shape[0],)),
    #         self.is_training: False,
    #     }
    #     z_mu, z_log_sigma_sq = sess.run([self.z_mu_pr, self.z_log_sigma_sq_pr], feed_dict=feed_dict)
    #     z_sigma = np.exp(0.5*z_log_sigma_sq)
    #     z_pr = np.random.normal(loc=z_mu, scale=z_sigma)
    #     feed_dict.update({
    #         self.use_z_ph: True,
    #         self.z_ph: z_pr,
    #     })
    #     preds= sess.run(self.y_hat, feed_dict=feed_dict)
    #     return preds
    #
    # def manipulate_z(self, sess, z_value, X_t_value):
    #     feed_dict = {
    #         self.use_z_ph: True,
    #         self.z_ph: z_value,
    #         self.X_t: X_t_value,
    #         self.is_training: False,
    #     }
    #     preds= sess.run(self.y_hat, feed_dict=feed_dict)
    #     return preds
    #
    # def compute_loss(self, sess, X_c_value, y_c_value, X_t_value, y_t_value, is_training):
    #     feed_dict = {
    #         self.X_c: X_c_value,
    #         self.y_c: y_c_value,
    #         self.X_t: X_t_value,
    #         self.y_t: y_t_value,
    #         self.is_training: is_training,
    #     }
    #     l = sess.run(self.loss, feed_dict=feed_dict)
    #     return l




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

            #
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

            for _ in range(4):
                outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
            outputs = tf.reshape(outputs, [-1, np.prod(int_shape(outputs)[1:])])
            z = tf.tile(z, tf.stack([bsize, 1]))
            outputs = tf.concat([outputs, z], axis=-1)

            outputs = dense(outputs, num_filters)
            y = dense(outputs, num_classes, nonlinearity=None, bn=False)
            return y


            # z_tile = tf.tile(tf.reshape(z, [1, 1, 1, int_shape(z)[-1]]), tf.stack([bsize, 28, 28, 1]))
            # outputs = tf.concat([outputs, z_tile], axis=-1)
            # outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
            # #
            # z_tile = tf.tile(tf.reshape(z, [1, 1, 1, int_shape(z)[-1]]), tf.stack([bsize, 14, 14, 1]))
            # outputs = tf.concat([outputs, z_tile], axis=-1)
            # outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
            # #
            # z_tile = tf.tile(tf.reshape(z, [1, 1, 1, int_shape(z)[-1]]), tf.stack([bsize, 7, 7, 1]))
            # outputs = tf.concat([outputs, z_tile], axis=-1)
            # outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
            # #
            # z_tile = tf.tile(tf.reshape(z, [1, 1, 1, int_shape(z)[-1]]), tf.stack([bsize, 4, 4, 1]))
            # outputs = tf.concat([outputs, z_tile], axis=-1)
            # outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
            # #
            # outputs = tf.reduce_mean(outputs, [1, 2])
            # outputs = tf.reshape(outputs, [-1, num_filters])
            # y = dense(outputs, num_classes, nonlinearity=None, bn=False)
            # return y


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
def cls_aggregator(r, y, z_dim, method=tf.reduce_mean, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("cls_aggregator", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
            r_dim = int_shape(r)[-1]
            r_arr = []
            for k in range(int_shape(y)[-1]):
                r_arr.append(method(tf.tile(y[:, k:k+1], [1,r_dim]) * r, axis=0, keepdims=True))
            r = tf.concat(r_arr, axis=-1)
            return r


@add_arg_scope
def conditional_decoder(x, z, num_classes=1, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("conditional_decoder", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
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
