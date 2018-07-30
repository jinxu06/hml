import sys
import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from .meta_learner import MetaLearner, sort_x

class NPLearner(MetaLearner):

    def __init__(self, session, parallel_models, optimize_op, train_set=None, eval_set=None, variables=None, lr=0.001, device_type='gpu', tags=["test"]):
        super().__init__(session, parallel_models, optimize_op, train_set, eval_set, variables, lr, device_type, tags)
        self.checkpoint_dir = "/data/ziz/jxu/neural_processes/" + self.save_dir
        self.result_dir = "results/neural_processes/" + self.save_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

    def visualise(self, save_name):
        pass

    def run_train(self):
        pass

    def run_eval(self):
        pass
