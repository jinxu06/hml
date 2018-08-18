import matplotlib
matplotlib.use('Agg')
import os
import sys
import json
import argparse
import time
from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from args import argument_parser, prepare_args
from data.load_data import load
from models.neural_processes import NeuralProcess
from learners.np_learner import NPLearner
from models.neural_processes import fc_encoder, aggregator, conditional_decoder

parser = argument_parser()
args = parser.parse_args()
args = prepare_args(args)

checkpoint_dir = "/data/ziz/jxu"
result_dir = "results"

train_set, val_set = load(dataset_name=args.dataset_name)

print(train_set)
tasks = train_set.sample(5)
print(tasks)
t = tasks[0]
a, b, c, d = t.sample(5, 5)
print(a)
print(b)
print(c)
print(d)

quit()

models = [NeuralProcess(counters={}, user_mode=args.user_mode) for i in range(args.nr_model)]

model_opt = {
    "sample_encoder": fc_encoder,
    "aggregator": aggregator,
    "conditional_decoder": conditional_decoder,
    "task_type": "regression",
    "obs_shape": [1],
    "r_dim": 128,
    "z_dim": 32,
    "nonlinearity": tf.nn.relu,
    "bn": False,
    "kernel_initializer": tf.contrib.layers.xavier_initializer(uniform=False),
    "kernel_regularizer":None,
}

model = tf.make_template('model', NeuralProcess.construct)

for i in range(args.nr_model):
    with tf.device('/'+ args.device_type +':%d' % (i%args.nr_gpu)):
        model(models[i], **model_opt)


tags = ["2d", "face", "regression"]
# save_dir = "/data/ziz/jxu/neural_processes/test-{0}".format(args.dataset_name)
learner = NPLearner(session=None, parallel_models=models, optimize_op=None, train_set=train_set, eval_set=val_set, variables=tf.trainable_variables(), lr=args.learning_rate, device_type=args.device_type, tags=tags, cdir=checkpoint_dir, rdir=result_dir)


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
        "gen_num_shots": partial(np.random.randint, low=1, high=20),
        "gen_test_shots": partial(np.random.randint, low=1, high=20),
        "load_params": args.load_params,
    }

    if args.user_mode == 'train':
        learner.run_train(**run_params)
    elif args.user_mode == 'eval':
        learner.run_eval(run_params["eval_samples"], num_shots=1, test_shots=50)
        learner.run_eval(run_params["eval_samples"], num_shots=5, test_shots=50)
        learner.run_eval(run_params["eval_samples"], num_shots=10, test_shots=50)
