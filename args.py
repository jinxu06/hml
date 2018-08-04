import os
import json
import argparse
from functools import partial
import numpy as np
import tensorflow as tf

def argument_parser():
    """
    Get an argument parser for a training script.
    """
    print("Construct argument parser ......")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pretrained', help='evaluate a pre-trained model',
                        action='store_true', default=False)
    parser.add_argument('--seed', help='random seed', default=0, type=int)
    parser.add_argument('--device_type', help='gpu or cpu', default='gpu')
    parser.add_argument('--gpus', help='IDs of GPUs used', default='0')
    parser.add_argument('--output_dir', help='output directory', default='')
    parser.add_argument('--checkpoint_dir', help='checkpoint directory', default='')
    parser.add_argument('--user_mode', help='user mode', default='train') # train | eval

    parser.add_argument('--debug', help='', action='store_true', default=False)
    parser.add_argument('--save_interval', type=int, default=10, help='Every how many epochs to write checkpoint/samples?')
    parser.add_argument('--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
    parser.add_argument('--dataset_name', help='name of dataset', default='gpsamples')
    #
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Base learning rate')
    parser.add_argument('--alpha', type=float, default=0.1, help='Inner step size')
    parser.add_argument('--nr_model', type=int, default=1, help='How many models are there with shared parameters?')
    parser.add_argument('--img_size', type=int, default=28, help="size of input image")
    parser.add_argument('--num_shots', type=int, default=5, help="")
    parser.add_argument('--test_shots', type=int, default=5, help="")
    parser.add_argument('--num_classes', type=int, default=5, help="")
    # parser.add_argument('--batch_size', type=int, default=100, help='Batch size during training per GPU')
    #
    parser.add_argument('--inner_batch', help='inner batch size', default=5, type=int)
    parser.add_argument('--inner_iters', help='inner iterations', default=20, type=int)
    parser.add_argument('--meta_step', help='meta-training step size', default=0.1, type=float)
    parser.add_argument('--meta_step_final', help='meta-training step size by the end',
                        default=0.1, type=float)
    parser.add_argument('--meta_batch', help='meta-training batch size', default=1, type=int)
    parser.add_argument('--meta_iters', help='meta-training iterations', default=400000, type=int)
    parser.add_argument('--eval_batch', help='eval inner batch size', default=5, type=int)
    parser.add_argument('--eval_iters', help='eval inner iterations', default=50, type=int)
    parser.add_argument('--eval_samples', help='evaluation samples', default=10000, type=int)
    parser.add_argument('--eval_interval', help='train steps per eval', default=10, type=int)
    return parser

def prepare_args(args):
    print("Prepare args ......")
    print("\t* infer nr_gpu")
    args.nr_gpu = len(args.gpus.split(","))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print("\t* check/make checkpoint directory")
    if not os.path.exists(args.checkpoint_dir) and args.checkpoint_dir!="":
        os.makedirs(args.checkpoint_dir)
    print("\t* check/make output directory")
    if not os.path.exists(args.output_dir) and args.output_dir!="":
        os.makedirs(args.output_dir)
    print("\t* set random seed {0} for numpy and tensorflow".format(args.seed))
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    print('Input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args
    return args


def model_kwards(model_name, parse_args, user_set_args={}):
    if model_name == 'omniglot':
        params = {
            "img_size": parse_args.img_size,
            "batch_size": parse_args.inner_batch,
            "nr_resnet": 3,
            "nr_filters": 30,
            "nonlinearity": tf.nn.elu,
            "bn": False,
            "kernel_initializer": tf.contrib.layers.xavier_initializer(),
            "kernel_regularizer":None,
        }
        params.update(user_set_args)
        return params


def learn_kwards(model_name, parse_args, user_set_args={}):
    if model_name == 'omniglot':
        params = {
            "eval_num_tasks": 20,
            "meta_iter_per_epoch": 100,
            "meta_batch_size": 5,
            "meta_step_size": 1e-4,
            "num_shots": 10,
            "test_shots": 10,
            "inner_iter": 5,
            "inner_batch_size": args.inner_batch,
        }
        params.update(user_set_args)
        return params


def train_kwargs(parsed_args):
    """
    Build kwargs for the train() function from the parsed
    command-line arguments.
    """
    return {
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'train_shots': (parsed_args.train_shots or None),
        'inner_batch_size': parsed_args.inner_batch,
        'inner_iters': parsed_args.inner_iters,
        'replacement': parsed_args.replacement,
        'meta_step_size': parsed_args.meta_step,
        'meta_step_size_final': parsed_args.meta_step_final,
        'meta_batch_size': parsed_args.meta_batch,
        'meta_iters': parsed_args.meta_iters,
        'eval_inner_batch_size': parsed_args.eval_batch,
        'eval_inner_iters': parsed_args.eval_iters,
        'eval_interval': parsed_args.eval_interval,
        'weight_decay_rate': parsed_args.weight_decay,
        'transductive': parsed_args.transductive,
    }

def evaluate_kwargs(parsed_args):
    """
    Build kwargs for the evaluate() function from the
    parsed command-line arguments.
    """
    return {
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'eval_inner_batch_size': parsed_args.eval_batch,
        'eval_inner_iters': parsed_args.eval_iters,
        'replacement': parsed_args.replacement,
        'weight_decay_rate': parsed_args.weight_decay,
        'num_samples': parsed_args.eval_samples,
        'transductive': parsed_args.transductive,
    }
