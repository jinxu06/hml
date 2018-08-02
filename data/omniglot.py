"""
Loading and augmenting the Omniglot dataset.
To use these APIs, you should prepare a directory that
contains all of the alphabets from both images_background
and images_evaluation.
"""

import os
import random

from PIL import Image
import numpy as np
import tensorflow as tf
import misc.helpers as helpers

def load_omniglot(data_dir, num_train=1200, augment_train_set=True):
    data = read_dataset(data_dir)
    train_set, eval_set = split_dataset(data)
    train_set = list(augment_dataset(train_set))
    eval_set = list(eval_set)
    return train_set, eval_set


class Omniglot(object):

    def __init__(self, chars, num_classes, dataset_name="omniglot"):
        self.dataset_name = dataset_name
        self.chars = chars
        self.num_char = len(chars)
        self.num_classes = num_classes

    def sample(self, num):
        tasks = []
        for _ in range(num):
            idx = np.random.choice(self.num_char, size=self.num_classes).astype(np.int32)
            tasks.append(Characters([self.chars[i] for i in idx]))
        return tasks


def read_dataset(data_dir):
    """
    Iterate over the characters in a data directory.
    Args:
      data_dir: a directory of alphabet directories.
    Returns:
      An iterable over Characters.
    The dataset is unaugmented and not split up into
    training and test sets.
    """
    for alphabet_name in sorted(os.listdir(data_dir)):
        alphabet_dir = os.path.join(data_dir, alphabet_name)
        if not os.path.isdir(alphabet_dir):
            continue
        for char_name in sorted(os.listdir(alphabet_dir)):
            if not char_name.startswith('character'):
                continue
            yield Character(os.path.join(alphabet_dir, char_name), 0)

def split_dataset(dataset, num_train=1200):
    """
    Split the dataset into a training and test set.
    Args:
      dataset: an iterable of Characters.
    Returns:
      A tuple (train, test) of Character sequences.
    """
    all_data = list(dataset)
    random.shuffle(all_data)
    return all_data[:num_train], all_data[num_train:]

def augment_dataset(dataset):
    """
    Augment the dataset by adding 90 degree rotations.
    Args:
      dataset: an iterable of Characters.
    Returns:
      An iterable of augmented Characters.
    """
    for character in dataset:
        for rotation in [0, 90, 180, 270]:
            yield Character(character.dir_path, rotation=rotation)

class Characters:

    def __init__(self, chars, one_hot=True):
        self.chars = chars
        self.num_char = len(chars)
        self.one_hot = one_hot

    def sample(self, num_shots, test_shots):
        total_shots = num_shots + test_shots
        assert total_shots <= 20, "num_shots+test_shots={0}, but only have 20 instances in each class".format(total_shots)
        xs_train, xs_test = [], []
        ys_train, ys_test = [], []
        for i, char in enumerate(self.chars):
            s = char.sample(total_shots)
            xs_train.append(s[:num_shots])
            xs_test.append(s[num_shots:])
            ys_train.append(np.ones(num_shots)*i)
            ys_test.append(np.ones(test_shots)*i)
        xs_train = np.concatenate(xs_train, axis=0)
        xs_test = np.concatenate(xs_test, axis=0)
        ys_train = np.concatenate(ys_train, axis=0)
        ys_test = np.concatenate(ys_test, axis=0)
        if self.one_hot:
            ys_train = helpers.one_hot(ys_train, self.num_char)
            ys_test = helpers.one_hot(ys_test, self.num_char)
        return xs_train, ys_train, xs_test, ys_test



class Character:
    """
    A single character class.
    """
    def __init__(self, dir_path, rotation=0):
        self.dir_path = dir_path
        self.rotation = rotation
        self._cache = {}

    def sample(self, num_images):
        """
        Sample images (as numpy arrays) from the class.
        Returns:
          A sequence of 28x28 numpy arrays.
          Each pixel ranges from 0 to 1.
        """
        names = [f for f in os.listdir(self.dir_path) if f.endswith('.png')]
        random.shuffle(names)
        images = []
        for name in names[:num_images]:
            images.append(self._read_image(os.path.join(self.dir_path, name)))
        return images

    def _read_image(self, path):
        if path in self._cache:
            return self._cache[path]
        with open(path, 'rb') as in_file:
            img = Image.open(in_file).resize((28, 28)).rotate(self.rotation)
            self._cache[path] = np.array(img).astype('float32')
            return self._cache[path]
