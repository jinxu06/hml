import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from misc.layers import conv2d, deconv2d, dense
from misc.helpers import int_shape, get_name, get_trainable_variables
from misc.metrics import accuracy

class MAMLRegressor(object):

    def __init__(self, counters={}, user_mode='train'):
        self.counters = counters
        self.user_mode = user_mode

    def construct(self, regressor, task_type, obs_shape, label_shape=[], num_classes=1, alpha=0.01, inner_iters=1, eval_iters=10, nonlinearity=tf.nn.relu, bn=False, kernel_initializer=None, kernel_regularizer=None):

        self.regressor = regressor
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
        self.label_shape = label_shape
        self.num_classes = num_classes
        self.alpha = alpha
        self.inner_iters = inner_iters
        self.eval_iters = eval_iters
        self.nonlinearity = nonlinearity
        self.bn = bn
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

        self.X_c = tf.placeholder(tf.float32, shape=tuple([None,]+obs_shape))
        self.y_c = tf.placeholder(tf.float32, shape=tuple([None,]+label_shape))
        self.X_t = tf.placeholder(tf.float32, shape=tuple([None,]+obs_shape))
        self.y_t = tf.placeholder(tf.float32, shape=tuple([None,]+label_shape))
        self.is_training = tf.placeholder(tf.bool, shape=())

        self._model()
        self.loss = self._loss()
        self.grads = tf.gradients(self.loss, tf.trainable_variables(), colocate_gradients_with_ops=True)

    def _model(self):
        default_args = {
            "nonlinearity": self.nonlinearity,
            "bn": self.bn,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "is_training": self.is_training,
            "counters": {}, #self.counters,
            "num_classes": self.num_classes,
        }
        self.outputs_sqs = []
        with arg_scope([self.regressor], **default_args):
            self.scope_name = get_name("maml_regressor", self.counters)
            with tf.variable_scope(self.scope_name):
                outputs = self.regressor(self.X_c)
                vars = get_trainable_variables([self.scope_name])
                vars = [v for v in vars if 'BatchNorm' not in v.name]
                self.vars = vars

                self.outputs_sqs.append(self.regressor(self.X_t, params=vars.copy()))
                for k in range(1, max(self.inner_iters, self.eval_iters)+1):
                    loss = self.error_func(self.y_c, outputs)
                    grads = tf.gradients(loss, vars, colocate_gradients_with_ops=True)
                    vars = [v - self.alpha * g for v, g in zip(vars, grads)]
                    outputs = self.regressor(self.X_c, params=vars.copy())
                    outputs_t = self.regressor(self.X_t, params=vars.copy())
                    self.outputs_sqs.append(outputs_t)

                self.y_hat_sqs = [self.pred_func(o) for o in self.outputs_sqs]
                self.loss_sqs = [self.error_func(self.y_t, o) for o in self.outputs_sqs]
                if self.task_type == 'classification':
                    self.acc_sqs = [accuracy(self.y_t, y_hat) for y_hat in self.y_hat_sqs]

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
        if self.task_type == 'classification':
            return [self.loss_sqs[step], self.acc_sqs[step]], feed_dict
        return [self.loss_sqs[step]], feed_dict


    # def predict(self, sess, X_c_value, y_c_value, X_t_value, step=None):
    #     feed_dict = {
    #         self.X_c: X_c_value,
    #         self.y_c: y_c_value,
    #         self.X_t: X_t_value,
    #         # self.y_t: np.zeros((X_t_value.shape[0],)),
    #         self.is_training: False,
    #     }
    #     if step is None:
    #         preds= sess.run(self.y_hat, feed_dict=feed_dict)
    #     else:
    #         preds= sess.run(self.eval_y_hats[step], feed_dict=feed_dict)
    #     return preds
    #
    #
    # def compute_loss(self, sess, X_c_value, y_c_value, X_t_value, y_t_value, is_training, step=None):
    #     feed_dict = {
    #         self.X_c: X_c_value,
    #         self.y_c: y_c_value,
    #         self.X_t: X_t_value,
    #         self.y_t: y_t_value,
    #         self.is_training: is_training,
    #     }
    #     if step is None:
    #         l = sess.run(self.loss, feed_dict=feed_dict)
    #     else:
    #         l = sess.run(self.losses[step], feed_dict=feed_dict)
    #     return l
    #
    # def compute_acc(self, sess, X_c_value, y_c_value, X_t_value, y_t_value, is_training, step=None):
    #     feed_dict = {
    #         self.X_c: X_c_value,
    #         self.y_c: y_c_value,
    #         self.X_t: X_t_value,
    #         self.y_t: y_t_value,
    #         self.is_training: is_training,
    #     }
    #     if step is None:
    #         l = sess.run(self.acc, feed_dict=feed_dict)
    #     else:
    #         l = sess.run(self.accs[step], feed_dict=feed_dict)
    #     return l



@add_arg_scope
def mlp5(X, params=None, num_classes=1, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
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
def mlp2(X, params=None, num_classes=1, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
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
def omniglot_conv(X, params=None, num_classes=1, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    # name = get_name("omniglot_conv", counters)
    name = "omniglot_conv"
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
        num_filters = 64
        filter_size = [3, 3]
        stride = [2, 2]
        with arg_scope([conv2d, dense], **default_args):
            outputs = X
            if params is None:
                outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
                outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
                outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
                outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
                outputs = tf.reduce_mean(outputs, [1, 2])
                outputs = tf.reshape(outputs, [-1, num_filters])
                y = dense(outputs, num_classes, nonlinearity=None, bn=False)
            else:
                outputs = conv2d(outputs, num_filters, W=params.pop(), b=params.pop(), filter_size=filter_size, stride=stride, pad="SAME")
                outputs = conv2d(outputs, num_filters, W=params.pop(), b=params.pop(), filter_size=filter_size, stride=stride, pad="SAME")
                outputs = conv2d(outputs, num_filters, W=params.pop(), b=params.pop(), filter_size=filter_size, stride=stride, pad="SAME")
                outputs = conv2d(outputs, num_filters, W=params.pop(), b=params.pop(), filter_size=filter_size, stride=stride, pad="SAME")
                outputs = tf.reduce_mean(outputs, [1, 2])
                outputs = tf.reshape(outputs, [-1, num_filters])
                y = dense(outputs, num_classes, W=params.pop(), b=params.pop(), nonlinearity=None, bn=False)
                assert len(params)==0, "{0}: feed-in parameter list is not empty".format(name)
            return y


@add_arg_scope
def miniimagenet_conv(X, params=None, num_classes=1, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("miniimagenet_conv", counters)
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
        num_filters = 32
        with arg_scope([conv2d, dense], **default_args):
            outputs = X
            if params is None:
                outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
                outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
                outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
                outputs = conv2d(outputs, num_filters, filter_size=filter_size, stride=stride, pad="SAME")
                outputs = tf.reduce_mean(outputs, [1, 2])
                outputs = tf.reshape(outputs, [-1, num_filters])
                y = dense(outputs, num_classes, nonlinearity=None, bn=False)
            else:
                outputs = conv2d(outputs, num_filters, W=params.pop(), b=params.pop(), filter_size=filter_size, stride=stride, pad="SAME")
                outputs = conv2d(outputs, num_filters, W=params.pop(), b=params.pop(), filter_size=filter_size, stride=stride, pad="SAME")
                outputs = conv2d(outputs, num_filters, W=params.pop(), b=params.pop(), filter_size=filter_size, stride=stride, pad="SAME")
                outputs = conv2d(outputs, num_filters, W=params.pop(), b=params.pop(), filter_size=filter_size, stride=stride, pad="SAME")
                outputs = tf.reduce_mean(outputs, [1, 2])
                outputs = tf.reshape(outputs, [-1, num_filters])
                y = dense(outputs, num_classes, W=params.pop(), b=params.pop(), nonlinearity=None, bn=False)
                assert len(params)==0, "{0}: feed-in parameter list is not empty".format(name)
            return y
