import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from blocks.layers import conv2d, deconv2d, dense, nin, gated_resnet
from blocks.layers import up_shifted_conv2d, up_left_shifted_conv2d, up_shift, left_shift
from blocks.layers import down_shifted_conv2d, down_right_shifted_conv2d, down_shift, right_shift, down_shifted_deconv2d, down_right_shifted_deconv2d
from blocks.losses import bernoulli_loss
from blocks.samplers import gaussian_sampler, mix_logistic_sampler, bernoulli_sampler
from blocks.helpers import int_shape, broadcast_masks_tf
from blocks.estimators import compute_2gaussian_kld



class NeuralProcess(object):

    def __init__(self, counters={}, user_mode='train'):
        self.counters = counters
        self.user_mode = user_mode

    def construct(self, sample_encoder, aggregator, conditional_decoder, obs_shape, r_dim, z_dim, nonlinearity=tf.nn.relu, bn=False, kernel_initializer=None, kernel_regularizer=None):
        #
        self.sample_encoder = sample_encoder
        self.aggregator = aggregator
        self.conditional_decoder = conditional_decoder
        self.obs_shape = obs_shape
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.nonlinearity = nonlinearity
        self.bn = bn
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        #
        self.X_c = tf.placeholder(tf.float32, shape=tuple([None,]+obs_shape))
        self.y_c = tf.placeholder(tf.float32, shape=(None,))
        self.X_t = tf.placeholder(tf.float32, shape=tuple([None,]+obs_shape))
        self.y_t = tf.placeholder(tf.float32, shape=(None,))
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.use_z_ph = tf.cast(tf.placeholder_with_default(False, shape=()), dtype=tf.float32)
        self.z_ph = tf.placeholder_with_default(np.zeros((1, self.z_dim), dtype=np.float32), shape=(1, self.z_dim))
        #
        self.outputs = self._model()
        self.y_hat = self.outputs
        self.loss = self._loss(beta=1.0, y_sigma=0.2)
        #
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
        with arg_scope([self.conditional_decoder], **default_args):
            default_args.update({"bn":False})
            with arg_scope([self.sample_encoder, self.aggregator], **default_args):
                self.scope_name = get_name("neural_process", self.counters)
                with tf.variable_scope(self.scope_name):
                    num_c = tf.shape(self.X_c)[0]
                    X_ct = tf.concat([self.X_c, self.X_t], axis=0)
                    y_ct = tf.concat([self.y_c, self.y_t], axis=0)
                    r_ct = self.sample_encoder(X_ct, y_ct, self.r_dim)
                    self.z_mu_pr, self.z_log_sigma_sq_pr, self.z_mu_pos, self.z_log_sigma_sq_pos = self.aggregator(r_ct, num_c, self.z_dim)
                    if self.user_mode == 'train':
                        z = gaussian_sampler(self.z_mu_pos, tf.exp(0.5*self.z_log_sigma_sq_pos))
                    elif self.user_mode == 'eval':
                        z = self.z_mu_pos
                    else:
                        raise Exception("unknown user_mode")
                    z = (1-self.use_z_ph) * z + self.use_z_ph * self.z_ph
                    y_hat = self.conditional_decoder(self.X_t, z)
                    return y_hat

    def _loss(self, beta=1., y_sigma=1./np.sqrt(2)):
        self.reg = compute_2gaussian_kld(self.z_mu_pr, self.z_log_sigma_sq_pr, self.z_mu_pos, self.z_log_sigma_sq_pos)
        self.nll = mean_squared_error(self.y_t, self.y_hat, sigma=y_sigma)
        return self.nll + beta * self.reg

    def predict(self, sess, X_c_value, y_c_value, X_t_value):
        feed_dict = {
            self.X_c: X_c_value,
            self.y_c: y_c_value,
            self.X_t: X_t_value,
            self.y_t: np.zeros((X_t_value.shape[0],)),
            self.is_training: False,
        }
        z_mu, z_log_sigma_sq = sess.run([self.z_mu_pr, self.z_log_sigma_sq_pr], feed_dict=feed_dict)
        z_sigma = np.exp(0.5*z_log_sigma_sq)
        z_pr = np.random.normal(loc=z_mu, scale=z_sigma)
        feed_dict.update({
            self.use_z_ph: True,
            self.z_ph: z_pr,
        })
        preds= sess.run(self.preds, feed_dict=feed_dict)
        return preds

    def manipulate_z(self, sess, z_value, X_t_value):
        feed_dict = {
            self.use_z_ph: True,
            self.z_ph: z_value,
            self.X_t: X_t_value,
            self.is_training: False,
        }
        preds= sess.run(self.preds, feed_dict=feed_dict)
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
