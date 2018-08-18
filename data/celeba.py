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
        elif which_set == 'valid':
            self.images = load(self.data_dir, subset=which_set, size=32, limit=200)[0]
        elif which_set == 'test':
            self.images = load(self.data_dir, subset=which_set, size=32, limit=200)[0]
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
        self.image = np.mean(self.image.astype(np.float32), axis=-1)
        print(self.image.shape)

    def sample(self, num_pixels):
        pass
