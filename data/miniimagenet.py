"""
Loading and using the Mini-ImageNet dataset.
To use these APIs, you should prepare a directory that
contains three sub-directories: train, test, and val.
Each of these three directories should contain one
sub-directory per WordNet ID.
"""

import os
import random

from PIL import Image
import numpy as np


class Miniimagenet(object):

    def __init__(self, data_dir, num_classes, which_set, dataset_name="miniimagenet"):
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.which_set = which_set
        self.dataset_name = dataset_name

        self.image_classes = read_dataset(self.data_dir, self.which_set)
        self.num_image_classes = len(self.image_classes)

    def sample(self, num):
        tasks = []
        for _ in range(num):
            idx = np.random.choice(self.num_image_classes, size=self.num_classes).astype(np.int32)
            tasks.append(ImageNetClasses([self.image_classes[i] for i in idx]))
        return tasks

def read_dataset(data_dir, which_set=None):

    if which_set is None:
        return tuple(_read_classes(os.path.join(data_dir, x)) for x in ['train', 'val', 'test'])
    else:
        return _read_classes(os.path.join(data_dir, which_set))

def _read_classes(dir_path):
    """
    Read the WNID directories in a directory.
    """
    return [ImageNetClass(os.path.join(dir_path, f)) for f in os.listdir(dir_path)
            if f.startswith('n')]


class ImageNetClasses:

    def __init__(self, image_classes, one_hot=True):
        self.image_classes = image_classes
        self.num_image_classes = len(image_classes)
        self.one_hot = one_hot

    def sample(self, num_shots, test_shots):
        total_shots = num_shots + test_shots
        assert total_shots <= 20, "num_shots+test_shots={0}, but only have 20 instances in each class".format(total_shots)
        xs_train, xs_test = [], []
        ys_train, ys_test = [], []
        for i, c in enumerate(self.image_classes):
            s = c.sample(total_shots)
            xs_train.append(s[:num_shots])
            xs_test.append(s[num_shots:])
            ys_train.append(np.ones(num_shots)*i)
            ys_test.append(np.ones(test_shots)*i)
        xs_train = np.concatenate(xs_train, axis=0)
        xs_test = np.concatenate(xs_test, axis=0)
        ys_train = np.concatenate(ys_train, axis=0)
        ys_test = np.concatenate(ys_test, axis=0)
        if self.one_hot:
            ys_train = helpers.one_hot(ys_train, self.num_image_classes)
            ys_test = helpers.one_hot(ys_test, self.num_image_classes)
        return xs_train, ys_train, xs_test, ys_test

# pylint: disable=R0903
class ImageNetClass:
    """
    A single image class.
    """
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self._cache = {}

    def sample(self, num_images):
        """
        Sample images (as numpy arrays) from the class.
        Returns:
          A sequence of 84x84x3 numpy arrays.
          Each pixel ranges from 0 to 1.
        """
        names = [f for f in os.listdir(self.dir_path) if f.endswith('.JPEG')]
        random.shuffle(names)
        images = []
        for name in names[:num_images]:
            images.append(self._read_image(name))
        return images

    def _read_image(self, name):
        if name in self._cache:
            return self._cache[name].astype('float32') / 0xff
        with open(os.path.join(self.dir_path, name), 'rb') as in_file:
            img = Image.open(in_file).resize((84, 84)).convert('RGB')
            self._cache[name] = np.array(img)
            return self._read_image(name)
