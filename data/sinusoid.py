import os
import random
import numpy as np


class Sinusoid(object):

    def __init__(self, amp_range, phase_range, period_range=[2*np.pi, 2*np.pi], input_range=[-5, 5], dataset_name="sinusoid"):
        self.dataset_name = dataset_name
        self.amp_range = amp_range
        self.phase_range = phase_range
        self.period_range = period_range
        self.input_range = input_range

    def sample(self, num):
        sines = []
        for i in range(num):
            amp = np.random.uniform(self.amp_range[0], self.amp_range[1])
            phase = np.random.uniform(self.phase_range[0], self.phase_range[1])
            period = np.random.uniform(self.period_range[0], self.period_range[1])
            sines.append(SineWave(amp, phase, period, self.input_range))
        return sines


class SineWave(object):
    """
    A single sine wave class.
    """
    def __init__(self, amp, phase, period, input_range):
        self.amp = amp
        self.phase = phase
        self.period = period
        self.input_range = input_range
        # self.tags = {"amp":amp, "phase":phase, "input_range":input_range}

    def query(self, X):
        y = self.amp * np.sin( 2*np.pi*(X - self.phase) / self.period )
        return y

    # def sample(self, num_samples):
    #     inputs = np.random.uniform(self.input_range[0], self.input_range[1], [num_samples,1])
    #     outputs = self.amp * np.sin( 2*np.pi*(inputs - self.phase) / self.period )
    #     samples = np.concatenate([inputs, outputs], axis=-1)
    #     np.random.shuffle(samples)
    #     return samples

    def sample(self, num_samples, xs=None):
        if xs is None:
            xs = np.random.uniform(self.input_range[0], self.input_range[1], [num_samples,1])
        ys = self.amp * np.sin( 2*np.pi*(xs[:, 0] - self.phase) / self.period )
        if xs is None:
            p = np.random.permutation(num_samples)
            xs = xs[p]
            ys = ys[p]
        return xs, ys

    def get_all_samples(self):
        return self.sample(200)
