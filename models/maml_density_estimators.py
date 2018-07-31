import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from misc.layers import conv2d, deconv2d, dense
from misc.layers import nin, gated_resnet
from misc.layers import up_shifted_conv2d, up_left_shifted_conv2d, up_shift, left_shift, down_shift, right_shift
from misc.layers import down_shifted_conv2d, down_right_shifted_conv2d, down_shifted_deconv2d, down_right_shifted_deconv2d
from misc.losses import bernoulli_loss
from misc.samplers import gaussian_sampler, bernoulli_sampler
from misc.helpers import int_shape, get_name, get_trainable_variables

class MAMLDensityEstimator(object):

    def __init__(self, counters={}, user_mode='train'):
        self.counters = counters
        self.user_mode = user_mode

    def construct(self, density_estimator, error_func, obs_shape, alpha=0.01, inner_iters=1, eval_iters=10, nonlinearity=tf.nn.relu, dropout_p=0.0, bn=False, kernel_initializer=None, kernel_regularizer=None):

        self.density_estimator = density_estimator
        self.error_func = error_func
        self.obs_shape = obs_shape
        self.alpha = alpha
        self.inner_iters = inner_iters
        self.eval_iters = eval_iters
        self.nonlinearity = nonlinearity
        self.dropout_p = dropout_p
        self.bn = bn
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

        self.X_c = tf.placeholder(tf.float32, shape=tuple([None,]+obs_shape))
        self.X_t = tf.placeholder(tf.float32, shape=tuple([None,]+obs_shape))
        self.is_training = tf.placeholder(tf.bool, shape=())

        self.outputs = self._model()
        self.y_hat = self._sample()
        self.loss = self._loss()

        self.grads = tf.gradients(self.loss, get_trainable_variables([self.scope_name]), colocate_gradients_with_ops=True)

    def _model(self):
        default_args = {
            "nonlinearity": self.nonlinearity,
            "dropout_p": self.dropout_p,
            "bn": self.bn,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "is_training": self.is_training,
            "counters": self.counters,
        }
        with arg_scope([self.density_estimator], **default_args):
            self.scope_name = get_name("maml_density_estimator", self.counters)
            with tf.variable_scope(self.scope_name):
                dist_params = self.density_estimator(self.X_c)
                vars = get_trainable_variables([self.scope_name])
                dist_params_arr = [self.density_estimator(self.X_t, params=vars.copy())]
                for k in range(1, max(self.inner_iters, self.eval_iters)+1):
                    loss = self.error_func(labels=self.y_c, predictions=y_hat)
                    grads = tf.gradients(loss, vars, colocate_gradients_with_ops=True)
                    vars = [v - self.alpha * g for v, g in zip(vars, grads)]
                    dist_params = self.density_estimator(self.X_c, params=vars.copy())
                    dist_params_t = self.density_estimator(self.X_t, params=vars.copy())
                    y_hat_t_arr.append(y_hat_t)
                self.eval_y_hats = y_hat_t_arr
                return y_hat_t_arr[self.inner_iters]

    def _loss(self):
        return self.error_func(self.X_t, self.outputs)


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


    def compute_loss(self, sess, X_c_value, y_c_value, X_t_value, y_t_value, is_training):
        feed_dict = {
            self.X_c: X_c_value,
            self.y_c: y_c_value,
            self.X_t: X_t_value,
            self.y_t: y_t_value,
            self.is_training: is_training,
        }
        l = sess.run(self.loss, feed_dict=feed_dict)
        return l



@add_arg_scope
def binary_pixelcnn(x, params=None, nr_resnet=1, nr_filters=100, nonlinearity=None, dropout_p=0.0, bn=False, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("binary_pixelcnn", counters)
    print("construct", name, "...")
    with arg_scope([gated_resnet], nonlinearity=nonlinearity, dropout_p=dropout_p):
        with arg_scope([gated_resnet, down_shifted_conv2d, down_right_shifted_conv2d, down_shifted_deconv2d, down_right_shifted_deconv2d], bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):

            xs = int_shape(x)
            #ap = tf.Variable(np.zeros((xs[1], xs[2], 1), dtype=np.float32), trainable=True)
            #aps = tf.stack([ap for _ in range(xs[0])], axis=0)
            x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], 3)

            if params is None:
                u_list = [down_shift(down_shifted_conv2d(
                    x_pad, num_filters=nr_filters, filter_size=[2, 3]))]  # stream for pixels above
                ul_list = [down_shift(down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1, 3])) +
                           right_shift(down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 1]))]  # stream for up and to the left

                for rep in range(nr_resnet):
                    u_list.append(gated_resnet(
                        u_list[-1], conv=down_shifted_conv2d))
                    ul_list.append(gated_resnet(
                        ul_list[-1], u_list[-1], conv=down_right_shifted_conv2d))

                u_list.append(down_shifted_conv2d(
                    u_list[-1], num_filters=nr_filters, strides=[2, 2]))
                ul_list.append(down_right_shifted_conv2d(
                    ul_list[-1], num_filters=nr_filters, strides=[2, 2]))

                for rep in range(nr_resnet):
                    u_list.append(gated_resnet(
                        u_list[-1], conv=down_shifted_conv2d))
                    ul_list.append(gated_resnet(
                        ul_list[-1], u_list[-1], conv=down_right_shifted_conv2d))

                u_list.append(down_shifted_conv2d(
                    u_list[-1], num_filters=nr_filters, strides=[2, 2]))
                ul_list.append(down_right_shifted_conv2d(
                    ul_list[-1], num_filters=nr_filters, strides=[2, 2]))

                for rep in range(nr_resnet):
                    u_list.append(gated_resnet(
                        u_list[-1], conv=down_shifted_conv2d))
                    ul_list.append(gated_resnet(
                        ul_list[-1], u_list[-1], conv=down_right_shifted_conv2d))

                # /////// down pass ////////

                u = u_list.pop()
                ul = ul_list.pop()
                for rep in range(nr_resnet):
                    u = gated_resnet(
                        u, u_list.pop(), conv=down_shifted_conv2d)
                    ul = gated_resnet(ul, tf.concat(
                        [u, ul_list.pop()], 3), conv=down_right_shifted_conv2d)

                u = down_shifted_deconv2d(
                    u, num_filters=nr_filters, strides=[2, 2])
                ul = down_right_shifted_deconv2d(
                    ul, num_filters=nr_filters, strides=[2, 2])

                for rep in range(nr_resnet + 1):
                    u = gated_resnet(
                        u, u_list.pop(), conv=down_shifted_conv2d)
                    ul = gated_resnet(ul, tf.concat(
                        [u, ul_list.pop()], 3), conv=down_right_shifted_conv2d)

                u = down_shifted_deconv2d(
                    u, num_filters=nr_filters, strides=[2, 2])
                ul = down_right_shifted_deconv2d(
                    ul, num_filters=nr_filters, strides=[2, 2])

                for rep in range(nr_resnet + 1):
                    u = gated_resnet(
                        u, u_list.pop(), conv=down_shifted_conv2d)
                    ul = gated_resnet(ul, tf.concat(
                        [u, ul_list.pop()], 3), conv=down_right_shifted_conv2d)

                x_out = nin(tf.nn.elu(ul), 1)

            else:

                u_list = [down_shift(down_shifted_conv2d(
                    x_pad, num_filters=nr_filters, W=params.pop(), b=params.pop(), filter_size=[2, 3]))]  # stream for pixels above
                ul_list = [down_shift(down_shifted_conv2d(x_pad, num_filters=nr_filters, W=params.pop(), b=params.pop(), filter_size=[1, 3])) +
                           right_shift(down_right_shifted_conv2d(x_pad, num_filters=nr_filters, W=params.pop(), b=params.pop(), filter_size=[2, 1]))]  # stream for up and to the left

                for rep in range(nr_resnet):
                    u_list.append(gated_resnet(
                        u_list[-1], params=params, conv=down_shifted_conv2d))
                    ul_list.append(gated_resnet(
                        ul_list[-1], u_list[-1], conv=down_right_shifted_conv2d))

                u_list.append(down_shifted_conv2d(
                    u_list[-1], num_filters=nr_filters, W=params.pop(), b=params.pop(), strides=[2, 2]))
                ul_list.append(down_right_shifted_conv2d(
                    ul_list[-1], num_filters=nr_filters, W=params.pop(), b=params.pop(), strides=[2, 2]))

                for rep in range(nr_resnet):
                    u_list.append(gated_resnet(
                        u_list[-1], params=params, conv=down_shifted_conv2d))
                    ul_list.append(gated_resnet(
                        ul_list[-1], u_list[-1], params=params, conv=down_right_shifted_conv2d))

                u_list.append(down_shifted_conv2d(
                    u_list[-1], num_filters=nr_filters, W=params.pop(), b=params.pop(), strides=[2, 2]))
                ul_list.append(down_right_shifted_conv2d(
                    ul_list[-1], num_filters=nr_filters, W=params.pop(), b=params.pop(), strides=[2, 2]))

                for rep in range(nr_resnet):
                    u_list.append(gated_resnet(
                        u_list[-1], params=params, conv=down_shifted_conv2d))
                    ul_list.append(gated_resnet(
                        ul_list[-1], u_list[-1], params=params, conv=down_right_shifted_conv2d))

                # /////// down pass ////////

                u = u_list.pop()
                ul = ul_list.pop()
                for rep in range(nr_resnet):
                    u = gated_resnet(
                        u, u_list.pop(), params=params, conv=down_shifted_conv2d)
                    ul = gated_resnet(ul, tf.concat(
                        [u, ul_list.pop()], 3), params=params, conv=down_right_shifted_conv2d)

                u = down_shifted_deconv2d(
                    u, num_filters=nr_filters, W=params.pop(), b=params.pop(), strides=[2, 2])
                ul = down_right_shifted_deconv2d(
                    ul, num_filters=nr_filters, W=params.pop(), b=params.pop(), strides=[2, 2])

                for rep in range(nr_resnet + 1):
                    u = gated_resnet(
                        u, u_list.pop(), params=params, conv=down_shifted_conv2d)
                    ul = gated_resnet(ul, tf.concat(
                        [u, ul_list.pop()], 3), params=params, conv=down_right_shifted_conv2d)

                u = down_shifted_deconv2d(
                    u, num_filters=nr_filters, W=params.pop(), b=params.pop(), strides=[2, 2])
                ul = down_right_shifted_deconv2d(
                    ul, num_filters=nr_filters, W=params.pop(), b=params.pop(), strides=[2, 2])

                for rep in range(nr_resnet + 1):
                    u = gated_resnet(
                        u, u_list.pop(), params=params, conv=down_shifted_conv2d)
                    ul = gated_resnet(ul, tf.concat(
                        [u, ul_list.pop()], 3), params=params, conv=down_right_shifted_conv2d)

                x_out = nin(tf.nn.elu(ul), 1, W=params.pop(), b=params.pop())


            assert len(u_list) == 0
            assert len(ul_list) == 0

            return x_out
