import sys
import random
import numpy as np
import tensorflow as tf
from learners.learner import Learner
from blocks.optimizers import adam_updates
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from .meta_learner import MetaLearner

class NPLearner(MetaLearner):

    def visualise(self, save_name):
        pass

    def run_train(self):
        pass

    def run_eval(self):
        pass
