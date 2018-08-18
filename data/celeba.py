import os
import sys
import random
from PIL import Image
import numpy as np
import tensorflow as tf
import misc.helpers as helpers


def read_imgs(dir, limit=-1):
    filenames = []
    for entry in os.scandir(dir):
        if limit==0:
            break
        if not entry.name.startswith('.') and entry.is_file():
            limit -= 1
            filenames.append(entry.name)
    filenames = sorted(filenames)
    imgs = np.array([np.array(Image.open(os.path.join(dir, filename))) for filename in filenames]).astype(np.uint8)
    return imgs


def load(data_dir="/data/ziz/not-backed-up/jxu/CelebA", subset='train', size=64, limit=-1):
    if subset in ['train', 'valid', 'test']:
        #trainx = np.load(os.path.join(data_dir, "img_cropped_celeba.npz"))['arr_0'][:200000, :, :, :]
        trainx = read_imgs(os.path.join(data_dir, "celeba{0}-{1}-new".format(size, subset)), limit=limit)
        trainy = np.ones((trainx.shape[0], ))
        return trainx, trainy
    else:
        raise NotImplementedError('subset should be either train, valid or test')


class CelebA(object):

    def __init__(self, data_dir, which_set, dataset_name="celeba"):
        self.data_dir = data_dir
        self.which_set = which_set
        self.dataset_name = dataset_name

        if which_set == 'train':
            self.images = load(self.data_dir, subset=which_set, size=32, limit=10000)[0]
        elif which_set == 'val':
            self.images = load(self.data_dir, subset="valid", size=32, limit=2000)[0]
        elif which_set == 'test':
            self.images = load(self.data_dir, subset=which_set, size=32, limit=500)[0]
        else:
            raise Exception("unknown {0}".format(which_set))

    def sample(self, num):
        tasks = []
        idxs = np.random.choice(self.images.shape[0], size=num, replace=False).astype(np.int32)
        for i in idxs:
            tasks.append(FaceCurve(self.images[i]))
        return tasks


class FaceCurve(object):

    def __init__(self, image):
        self.image = image
        self.image = np.mean(self.image.astype(np.float32), axis=-1) / 255.
        self.xs, self.ys = np.meshgrid(np.arange(32), np.arange(32))
        self.xs = self.xs.astype(np.int32) / 32.
        self.ys = self.ys.astype(np.int32) / 32.
        self.xs = np.ndarray.flatten(self.xs)
        self.ys = np.ndarray.flatten(self.ys)
        self.cs = np.stack([self.xs, self.ys], axis=-1)
        self.bs = np.ndarray.flatten(self.image)
        self.num_total_pixels = len(self.xs)

    def sample(self, num_shots, test_shots):
        idx = np.random.choice(self.num_total_pixels, size=num_shots+test_shots, replace=False).astype(np.int32)
        return self.cs[idx][:num_shots], self.bs[idx][:num_shots], self.cs[idx][num_shots:], self.bs[idx][num_shots:]

    def show(self, bs=None, cs=None):
        if bs is None:
            bs = self.bs
        if cs is None:
            cs = self.cs
        img = np.ones((32, 32))
        bs = (bs * 32.).astype(np.int32)
        for b, c in zip(bs, cs):
            img[b[1], b[0]] = c
        img = np.stack([img for i in range(3)], axis=-1)
        return img


    def get_all_samples(self):
        return self.sample(self.num_total_pixels, 0)[:2]
