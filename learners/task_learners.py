import os
import sys
import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from .meta_learner import MetaLearner, cosort_x
from .np_learner import NPLearner
from .maml_learner import MAMLLearner
from .gavi_learner import GAVILearner

class NP2DRegressionLearner(NPLearner):

    def __init__(self, session, parallel_models, optimize_op, train_set=None, eval_set=None, variables=None, lr=0.001, device_type='gpu', tags=["test"], cdir="", rdir=""):
        super().__init__(session, parallel_models, optimize_op, train_set, eval_set, variables, lr, device_type, tags, cdir, rdir)


    def visualise_2d(self, save_name):
        m = self.parallel_models[0]
        fig = plt.figure(figsize=(12, 12))
        sampler = self.eval_set.sample(1)[0]
        c = [15, 30, 90, 512, 1024]
        for i in range(5):
            num_shots = c[i]
            X_c_value, y_c_value, X_t_value, y_t_value = sampler.sample(num_shots, test_shots=32*32-num_shots)
            X_value = np.concatenate([X_c_value, X_t_value], axis=0)
            y_value = np.concatenate([y_c_value, y_t_value], axis=0)
            for j in range(5):
                idx = 5 * j + i + 1
                ax = fig.add_subplot(5, 5, idx)
                ax.grid(False)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

                if j == 0:
                    img = sampler.show(X_c_value, y_c_value)
                    ax.imshow(img)
                else:
                    ops, feed_dict = m.predict(X_c_value, y_c_value, X_value)
                    y_hat = self.session.run(ops, feed_dict=feed_dict)[0]
                    img = sampler.show(X_value, y_hat)
                    ax.imshow(img)

        fig.savefig(save_name)
        plt.close()


    def run_train(self, num_epoch, eval_interval, save_interval, eval_samples, meta_batch, gen_num_shots, gen_test_shots, load_params=False):
        saver = tf.train.Saver(var_list=self.variables)
        if load_params:
            ckpt_file = self.checkpoint_dir + '/params.ckpt'
            print('restoring parameters from', ckpt_file)
            saver.restore(self.session, ckpt_file)
        self.visualise_2d(os.path.join(self.result_dir, "{0}-{1}.pdf".format(self.eval_set.dataset_name, 0)))

        for epoch in range(1, num_epoch+1):
            self.qclock()
            for k in range(1000):
                self.train(meta_batch, gen_num_shots, gen_test_shots)
            train_time = self.qclock()
            print("Epoch {0}: {1:0.3f}s ...................".format(epoch, train_time))
            if epoch % eval_interval == 0:
                v = self.evaluate(eval_samples, gen_num_shots, gen_test_shots)
                print("    Eval Loss: ", v)
                # v = self.test(eval_samples, num_shots, test_shots)

            if epoch % save_interval == 0:
                print("\tsave figure")
                self.visualise_2d(os.path.join(self.result_dir, "{0}-{1}.pdf".format(self.eval_set.dataset_name, epoch)))
                print("\tsave checkpoint")
                saver.save(self.session, self.checkpoint_dir + '/params.ckpt')
            sys.stdout.flush()


class LDVI2DLearner(GAVILearner):

    def __init__(self, session, parallel_models, optimize_op, train_set=None, eval_set=None, variables=None, lr=0.001, device_type='gpu', tags=["test"], cdir="", rdir=""):
        super().__init__(session, parallel_models, optimize_op, train_set, eval_set, variables, lr, device_type, tags, cdir, rdir)
        self.checkpoint_dir = os.path.join(cdir, "ldvi_processes", self.save_dir)
        self.result_dir = os.path.join(rdir, "ldvi_processes", self.save_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)


    def visualise_2d(self, save_name):
        m = self.parallel_models[0]
        fig = plt.figure(figsize=(12, 12))
        sampler = self.eval_set.sample(1)[0]
        c = [15, 30, 90, 512, 1024]
        for i in range(5):
            num_shots = c[i]
            X_c_value, y_c_value, X_t_value, y_t_value = sampler.sample(num_shots, test_shots=32*32-num_shots)
            X_value = np.concatenate([X_c_value, X_t_value], axis=0)
            y_value = np.concatenate([y_c_value, y_t_value], axis=0)
            for j in range(5):
                idx = 5 * j + i + 1
                ax = fig.add_subplot(5, 5, idx)
                ax.grid(False)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

                if j == 0:
                    img = sampler.show(X_c_value, y_c_value)
                    ax.imshow(img)
                else:
                    ops, feed_dict = m.predict(X_c_value, y_c_value, X_value)
                    y_hat = self.session.run(ops, feed_dict=feed_dict)[0]
                    img = sampler.show(X_value, y_hat)
                    ax.imshow(img)

        fig.savefig(save_name)
        plt.close()

    def run_train(self, num_epoch, eval_interval, save_interval, eval_samples, meta_batch, gen_num_shots, gen_test_shots, load_params=False):
        saver = tf.train.Saver(var_list=self.variables)
        if load_params:
            ckpt_file = self.checkpoint_dir + '/params.ckpt'
            print('restoring parameters from', ckpt_file)
            saver.restore(self.session, ckpt_file)
        self.visualise_2d(os.path.join(self.result_dir, "{0}-{1}.pdf".format(self.eval_set.dataset_name, 0)))

        for epoch in range(1, num_epoch+1):
            self.qclock()
            for k in range(1000):
                self.train(meta_batch, gen_num_shots, gen_test_shots)
            train_time = self.qclock()
            print("Epoch {0}: {1:0.3f}s ...................".format(epoch, train_time))
            if epoch % eval_interval == 0:
                v = self.evaluate(eval_samples, gen_num_shots, gen_test_shots)
                print("    Eval Loss: ", v)

            if epoch % save_interval == 0:
                print("\tsave figure")
                self.visualise_2d(os.path.join(self.result_dir, "{0}-{1}.pdf".format(self.eval_set.dataset_name, epoch)))
                #print("\tsave checkpoint")
                #saver.save(self.session, self.checkpoint_dir + '/params.ckpt')
            sys.stdout.flush()
