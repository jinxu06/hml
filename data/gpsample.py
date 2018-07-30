import numpy as np
import tensorflow as tf

# refer to https://gist.github.com/neubig/e859ef0cc1a63d1c2ea4

class GPSample(object):

    def __init__(self, input_range, var_range, max_num_samples=200, data=None, dataset_name="gpsamples"):
        self.dataset_name = dataset_name
        self.input_range = input_range
        self.var_range = var_range
        self.max_num_samples = max_num_samples
        self.data = data
        if data is not None:
            self.num_samples = self.data['xs'].shape[0]

    def sample(self, num):
        if self.data is None:
            return [self._sample_function(self.max_num_samples) for i in range(num)]
        p = np.random.choice(self.num_samples, size=(num), replace=False)
        return [GPFunction(xs=self.data['xs'][i][:,0], ys=self.data['ys'][i]) for i in p]


    def _sample_function(self, num_samples):
        xs = np.random.uniform(low=self.input_range[0], high=self.input_range[1], size=num_samples)
        mean = [0 for x in xs]
        var = np.random.uniform(low=self.var_range[0], high=self.var_range[1])
        gram = gram_matrix(xs, variance=var)
        ys = np.random.multivariate_normal(mean, gram)
        return GPFunction(xs, ys)


def rbf_kernel(x1, x2, variance = 1):
    return np.exp(-1 * ((x1-x2) ** 2) / (2*variance))

def gram_matrix(xs, variance=1):
    return [[rbf_kernel(x1,x2, variance) for x2 in xs] for x1 in xs]


class GPFunction(object):

    def __init__(self, xs, ys):
        assert len(xs)==len(ys), "len(xs)!=len(ys)"
        self.xs = xs
        self.ys = ys
        self.num_samples = len(xs)

    def sample(self, num_samples):
        assert num_samples <= self.num_samples, "num_samples exceed max_num_samples"
        p = np.random.choice(self.num_samples, size=(num_samples,), replace=False)
        return self.xs[p][:,None], self.ys[p]

    def get_all_samples(self):
        return self.xs[:,None], self.ys
