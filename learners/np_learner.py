import os
import sys
import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from .meta_learner import MetaLearner, cosort_x

class NPLearner(MetaLearner):

    def __init__(self, session, parallel_models, optimize_op, train_set=None, eval_set=None, variables=None, lr=0.001, device_type='gpu', tags=["test"], cdir="", rdir=""):
        super().__init__(session, parallel_models, optimize_op, train_set, eval_set, variables, lr, device_type, tags)
        self.checkpoint_dir = os.path.join(cdir, "neural_processes", self.save_dir)
        self.result_dir = os.path.join(rdir, "neural_processes", self.save_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)


    def train(self, meta_batch, num_shots=None, test_shots=None):
        assert meta_batch==self.nr_model, "nr_model != meta_batch"
        tasks = self.train_set.sample(meta_batch)
        feed_dict = {}
        for i, task in enumerate(tasks):
            if num_shots is None:
                num_shots = np.random.randint(low=1, high=50)
            if test_shots is None:
                test_shots = 20

            X_c_value, y_c_value, X_t_value, y_t_value = task.sample(num_shots, test_shots)
            X_value = np.concatenate([X_c_value, X_t_value], axis=0)
            y_value = np.concatenate([y_c_value, y_t_value], axis=0)
            feed_dict.update({
                self.parallel_models[i].X_c: X_c_value,
                self.parallel_models[i].y_c: y_c_value,
                self.parallel_models[i].X_t: X_c_value,
                self.parallel_models[i].y_t: y_c_value,
                self.parallel_models[i].is_training: True,
            })
        self.get_session().run(self.optimize_op, feed_dict=feed_dict)

    def evaluate(self, eval_samples, num_shots=None, test_shots=None):
        evals = []
        eval_meta_batch = eval_samples // self.nr_model
        for i in range(eval_meta_batch):
            tasks = self.eval_set.sample(self.nr_model)
            if num_shots is None:
                num_shots = np.random.randint(low=1, high=50)
            if test_shots is None:
                test_shots = 20

            run_ops, feed_dict = [], {}
            for k, task in enumerate(tasks):
                X_c_value, y_c_value, X_t_value, y_t_value = task.sample(num_shots, test_shots)
                X_value = np.concatenate([X_c_value, X_t_value], axis=0)
                y_value = np.concatenate([y_c_value, y_t_value], axis=0)
                ## !! training data is not included in the evaluation, different from neural process
                ops, d = self.parallel_models[k].evaluate_metrics(X_c_value, y_c_value, X_c_value, y_c_value)
                run_ops += ops
                feed_dict.update(d)
            ls = np.array(self.get_session().run(run_ops, feed_dict=feed_dict))
            ls = np.reshape(ls, (self.nr_model, len(ls)//self.nr_model))
            ls = np.mean(ls, axis=0)
            evals.append(ls)
        return np.mean(evals, axis=0)


    def test(self, eval_samples, num_shots=None, test_shots=None):
        m = self.parallel_models[0]
        evals = []
        for i in range(eval_samples):
            task = self.eval_set.sample(1)[0]
            if num_shots is None:
                num_shots = np.random.randint(low=1, high=50)
            if test_shots is None:
                test_shots = 20

            X_c_value, y_c_value, X_t_value, y_t_value = task.sample(num_shots, test_shots)
            X_value = np.concatenate([X_c_value, X_t_value], axis=0)
            y_value = np.concatenate([y_c_value, y_t_value], axis=0)
            feed_dict = {}
            feed_dict.update({
                m.X_c: X_c_value,
                m.y_c: y_c_value,
                m.X_t: X_c_value,
                m.y_t: y_c_value,
                m.is_training: True,
            })

            v = self.get_session().run(m.r_ct[0], feed_dict=feed_dict)
            print(v)


    def visualise_1d(self, save_name):
        fig = plt.figure(figsize=(10, 10))
        for i in range(12):
            ax = fig.add_subplot(4, 3, i+1)
            sampler = self.eval_set.sample(1)[0]
            c = [1, 5, 10, 20]
            num_shots = c[(i%4)]
            X_c_value, y_c_value, X_t_value, y_t_value = sampler.sample(num_shots, test_shots=0)
            X_value = np.concatenate([X_c_value, X_t_value], axis=0)
            y_value = np.concatenate([y_c_value, y_t_value], axis=0)
            m = self.parallel_models[0]
            X_gt, y_gt = sampler.get_all_samples()
            ax.plot(*cosort_x(X_gt[:,0], y_gt), "-")
            ax.scatter(X_c_value[:,0], y_c_value)

            for k in range(20):
                X_eval = np.linspace(self.eval_set.input_range[0], self.eval_set.input_range[1], num=100)[:,None]
                y_hat = m.predict(self.session, X_c_value, y_c_value, X_eval)
                ax.plot(X_eval[:,0], y_hat, "-", color='gray', alpha=0.5)

        fig.savefig(save_name)
        plt.close()


    def run_train(self, num_epoch, eval_interval, save_interval, eval_samples, meta_batch, num_shots, test_shots, load_params=False):
        saver = tf.train.Saver(var_list=self.variables)
        if load_params:
            ckpt_file = self.checkpoint_dir + '/params.ckpt'
            print('restoring parameters from', ckpt_file)
            saver.restore(self.session, ckpt_file)
        # self.visualise_1d(os.path.join(self.result_dir, "{0}-{1}.pdf".format(self.eval_set.dataset_name, 0)))

        for epoch in range(1, num_epoch+1):
            self.qclock()
            for k in range(1000):
                self.train(meta_batch, num_shots, test_shots)
            train_time = self.qclock()
            print("Epoch {0}: {1:0.3f}s ...................".format(epoch, train_time))
            if epoch % eval_interval == 0:
                v = self.evaluate(eval_samples, num_shots, test_shots)
                print("    Eval Loss: ", v)
                v = self.test(eval_samples, num_shots, test_shots)

            if epoch % save_interval == 0:
                print("\tsave figure")
                # self.visualise_1d(os.path.join(self.result_dir, "{0}-{1}.pdf".format(self.eval_set.dataset_name, epoch)))
                print("\tsave checkpoint")
                saver.save(self.session, self.checkpoint_dir + '/params.ckpt')
            sys.stdout.flush()

    def run_eval(self, num_func, num_shots, test_shots):
        saver = tf.train.Saver(var_list=self.variables)
        ckpt_file = self.checkpoint_dir + '/params.ckpt'
        print('restoring parameters from', ckpt_file)
        saver.restore(self.session, ckpt_file)
        m = self.parallel_models[0]
        evals = []
        for _ in range(num_func):
            sampler = self.eval_set.sample(1)[0]
            X_c_value, y_c_value, X_t_value, y_t_value = sampler.sample(num_shots, test_shots)
            X_value = np.concatenate([X_c_value, X_t_value], axis=0)
            y_value = np.concatenate([y_c_value, y_t_value], axis=0)
            l = m.compute_loss(self.get_session(), X_c_value, y_c_value, X_value, y_value, is_training=False)
            evals.append(l)
        eval = np.nanmean(evals)
        print(".......... EVAL : num_func {0} num_shots {1} test_shots {2}  ............".format(num_func, num_shots, test_shots))
        print("\t{0}".format(eval))

        self.visualise_1d(os.path.join(self.result_dir, "{0}-{1}.pdf".format(self.eval_set.dataset_name, "eval")))
