import matplotlib
matplotlib.use('Agg')
import os
import sys
import json
import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from args import argument_parser, prepare_args
parser = argument_parser()
args = parser.parse_args()
args = prepare_args(args)

from data.load_data import load
train_set, val_set = load(dataset_name=args.dataset_name)

from models.maml_regressor import MAMLRegressor, mlp5
from learners.maml_learner import MAMLLearner
models = [MAMLRegressor(counters={}, user_mode=args.user_mode) for i in range(args.nr_model)]

model_opt = {
    "regressor": mlp5,
    "error_func": tf.losses.mean_squared_error,
    "obs_shape": [1],
    "alpha": 0.01,
    "nonlinearity": tf.nn.relu,
    "bn": False,
    "kernel_initializer": tf.contrib.layers.xavier_initializer(uniform=False),
    "kernel_regularizer":None,
}

model = tf.make_template('model', MAMLRegressor.construct)

for i in range(args.nr_model):
    with tf.device('/'+ args.device_type +':%d' % (i%args.nr_gpu)):
        model(models[i], **model_opt)

save_dir = "/data/ziz/jxu/maml/test-{0}".format(args.dataset_name)
learner = MAMLLearner(session=None, parallel_models=models, optimize_op=None, train_set=train_set, eval_set=val_set, variables=tf.trainable_variables(), lr=args.learning_rate, device_type=args.device_type, save_dir=save_dir)

initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    if args.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(initializer)

    learner.set_session(sess)

    run_params = {
        "num_epoch": 500,
        "eval_interval": 5,
        "save_interval": args.save_interval,
        "eval_samples": 1000,
        "meta_batch": args.nr_model,
        "num_shots": 10,
        "test_shots": 10,
        "load_params": args.load_params,
    }
    if args.user_mode == 'train':
        learner.run_train(**run_params)
    elif args.user_mode == 'eval':
        pass
