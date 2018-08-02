import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from misc.layers import conv2d, deconv2d, dense
from misc.helpers import int_shape, get_name, get_trainable_variables

class MAMLRegressor(object):

    def __init__(self, counters={}, user_mode='train'):
        self.counters = counters
        self.user_mode = user_mode

    def construct(self, regressor, error_func, obs_shape, alpha=0.01, inner_iters=1, eval_iters=10, nonlinearity=tf.nn.relu, bn=False, kernel_initializer=None, kernel_regularizer=None):

        self.regressor = regressor
        self.error_func = error_func
        self.obs_shape = obs_shape
        self.alpha = alpha
        self.inner_iters = inner_iters
        self.eval_iters = eval_iters
        self.nonlinearity = nonlinearity
        self.bn = bn
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

        self.X_c = tf.placeholder(tf.float32, shape=tuple([None,]+obs_shape))
        self.y_c = tf.placeholder(tf.float32, shape=(None,))
        self.X_t = tf.placeholder(tf.float32, shape=tuple([None,]+obs_shape))
        self.y_t = tf.placeholder(tf.float32, shape=(None,))
        self.is_training = tf.placeholder(tf.bool, shape=())

        self.outputs = self._model()
        self.y_hat = self.outputs
        self.loss = self._loss()

        self.grads = tf.gradients(self.loss, get_trainable_variables([self.scope_name]), colocate_gradients_with_ops=True)

    def _model(self):
        default_args = {
            "nonlinearity": self.nonlinearity,
            "bn": self.bn,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "is_training": self.is_training,
            "counters": self.counters,
        }
        with arg_scope([self.regressor], **default_args):
            self.scope_name = get_name("maml_regressor", self.counters)
            with tf.variable_scope(self.scope_name):
                y_hat = self.regressor(self.X_c)
                vars = get_trainable_variables([self.scope_name])
                y_hat_t_arr = [self.regressor(self.X_t, params=vars.copy())]
                for k in range(1, max(self.inner_iters, self.eval_iters)+1):
                    loss = self.error_func(labels=self.y_c, predictions=y_hat)
                    grads = tf.gradients(loss, vars, colocate_gradients_with_ops=True)
                    vars = [v - self.alpha * g for v, g in zip(vars, grads)]
                    y_hat = self.regressor(self.X_c, params=vars.copy())
                    y_hat_t = self.regressor(self.X_t, params=vars.copy())
                    y_hat_t_arr.append(y_hat_t)
                self.eval_y_hats = y_hat_t_arr
                return y_hat_t_arr[self.inner_iters]

    def _loss(self):
        self.losses = [self.error_func(labels=self.y_t, predictions=y_hat) for y_hat in self.eval_y_hats]
        return self.losses[1]
        #return self.error_func(labels=self.y_t, predictions=self.y_hat)


    def predict(self, sess, X_c_value, y_c_value, X_t_value, step=None):
        feed_dict = {
            self.X_c: X_c_value,
            self.y_c: y_c_value,
            self.X_t: X_t_value,
            # self.y_t: np.zeros((X_t_value.shape[0],)),
            self.is_training: False,
        }
        if step is None:
            preds= sess.run(self.y_hat, feed_dict=feed_dict)
        else:
            preds= sess.run(self.eval_y_hats[step], feed_dict=feed_dict)
        return preds


    def compute_loss(self, sess, X_c_value, y_c_value, X_t_value, y_t_value, is_training, step=None):
        feed_dict = {
            self.X_c: X_c_value,
            self.y_c: y_c_value,
            self.X_t: X_t_value,
            self.y_t: y_t_value,
            self.is_training: is_training,
        }
        if step is None:
            l = sess.run(self.loss, feed_dict=feed_dict)
        else:
            l = sess.run(self.losses[step], feed_dict=feed_dict)
        return l



@add_arg_scope
def mlp5(X, params=None, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("mlp5", counters)
    print("construct", name, "...")
    if params is not None:
        params.reverse()
    with tf.variable_scope(name):
        default_args = {
            "nonlinearity": nonlinearity,
            "bn": bn,
            "kernel_initializer": kernel_initializer,
            "kernel_regularizer": kernel_regularizer,
            "is_training": is_training,
            "counters": counters,
        }
        with arg_scope([dense], **default_args):
            batch_size = tf.shape(X)[0]
            size = 256
            outputs = X
            for k in range(4):
                if params is not None:
                    outputs = dense(outputs, size, W=params.pop(), b=params.pop())
                else:
                    outputs = dense(outputs, size)
            if params is not None:
                outputs = dense(outputs, 1, nonlinearity=None, W=params.pop(), b=params.pop())
            else:
                outputs = dense(outputs, 1, nonlinearity=None)
            outputs = tf.reshape(outputs, shape=(batch_size,))
            if params is not None:
                assert len(params)==0, "{0}: feed-in parameter list is not empty".format(name)
            return outputs


@add_arg_scope
def mlp2(X, params=None, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    "Replicate Finn's MAML paper"
    name = get_name("mlp2", counters)
    print("construct", name, "...")
    if params is not None:
        params.reverse()
    with tf.variable_scope(name):
        default_args = {
            "nonlinearity": nonlinearity,
            "bn": bn,
            "kernel_initializer": kernel_initializer,
            "kernel_regularizer": kernel_regularizer,
            "is_training": is_training,
            "counters": counters,
        }
        with arg_scope([dense], **default_args):
            batch_size = tf.shape(X)[0]
            size = 40
            outputs = X
            for k in range(2):
                if params is not None:
                    outputs = dense(outputs, size, W=params.pop(), b=params.pop())
                else:
                    outputs = dense(outputs, size)
            if params is not None:
                outputs = dense(outputs, 1, nonlinearity=None, W=params.pop(), b=params.pop())
            else:
                outputs = dense(outputs, 1, nonlinearity=None)
            outputs = tf.reshape(outputs, shape=(batch_size,))
            if params is not None:
                assert len(params)==0, "{0}: feed-in parameter list is not empty".format(name)
            return outputs


@add_arg_scope
def omniglot_conv(X, params=None, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("omniglot_conv", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([conv2d, dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
            outputs = X
            outputs = conv2d(outputs, 64, filter_size=3, stride=1, pad="SAME")
            outputs = conv2d(outputs, 64, 3, 2, "SAME")
            outputs = conv2d(outputs, 128, 3, 1, "SAME")
            outputs = conv2d(outputs, 128, 3, 2, "SAME")
            outputs = conv2d(outputs, 256, 4, 1, "VALID")
            outputs = conv2d(outputs, 256, 4, 1, "VALID")
            outputs = tf.reshape(outputs, [-1, 256])
            y = tf.dense(outputs, 1, nonlinearity=None, bn=False)
            return y
