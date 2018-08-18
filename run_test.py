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


from data.celeba import CelebA


celeba = CelebA(data_dir="/data/ziz/not-backed-up/jxu/CelebA", which_set='train')
pritn(celeba.sample(10))
