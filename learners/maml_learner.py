import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from .meta_learner import MetaLearner

class MAMLLearner(MetaLearner):

    def visualise(self, save_name, num_function, num_shots, test_shots, input_range=(-2., 2.)):
        fig = plt.figure(figsize=(10,10))
        # a = int(np.sqrt(num_function))
        for i in range(num_function):
            # ax = fig.add_subplot(a,a,i+1)
            ax = fig.add_subplot(4,3,i+1)
            sampler = self.eval_set.sample(1)[0]

            c = [1, 4, 8, 16, 32, 64]
            num_shots = c[(i%6)]

            X_value, y_value = sampler.sample(num_shots+test_shots)
            X_c_value, X_t_value = X_value[:num_shots], X_value[num_shots:]
            y_c_value, y_t_value = y_value[:num_shots], y_value[num_shots:]
            m = self.parallel_models[0]
            X_gt, y_gt = sampler.get_all_samples()
            ax.plot(*sort_x(X_gt[:,0], y_gt), "-")
            ax.scatter(X_c_value[:,0], y_c_value)


            X_eval = np.linspace(self.eval_set.input_range[0], self.eval_set.input_range[1], num=100)[:,None]
            # step 1
            y_hat = m.predict(self.session, X_c_value, y_c_value, X_eval, step=1)
            ax.plot(X_eval[:,0], y_hat, ":", color='gray', alpha=0.3)
            # step 5
            y_hat = m.predict(self.session, X_c_value, y_c_value, X_eval, step=5)
            ax.plot(X_eval[:,0], y_hat, "--", color='gray', alpha=0.3)
            # step 10
            y_hat = m.predict(self.session, X_c_value, y_c_value, X_eval, step=10)
            ax.plot(X_eval[:,0], y_hat, "-", color='gray', alpha=0.3)

        fig.savefig(save_name)
        plt.close()


    def run_train(self, num_epoch, eval_interval, save_interval, eval_samples, meta_batch, num_shots, test_shots, load_params=False):
        super().run_train(load_params=load_params)
        self.visualise("fig/test.pdf", num_figures, num_shots, test_shots)

        for epoch in range(1, num_epoch+1):
            self.qclock()
            for k in range(1000):
                self.train(meta_batch, num_shots, test_shots)
            train_time = self.qclock()
            print("Epoch {0}: {1:0.3f}s ...................".format(epoch, train_time))
            if epoch % eval_interval == 0:
                v = self.evaluate(eval_samples, num_shots, test_shots)
                print("    Eval Loss: ", v)
            if epoch % save_interval == 0:
                print("\tsave figure")
                self.visualise("fig/test.pdf", num_figures, num_shots, test_shots)
                print("\tsave checkpoint")
                saver.save(self.session, self.save_dir + '/params.ckpt')
            sys.stdout.flush()
