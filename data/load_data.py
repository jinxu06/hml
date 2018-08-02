import numpy as np


def load(dataset_name, **kwargs):
    if dataset_name == 'gpsamples':
        return load_gpsamples(**kwargs)
    elif dataset_name == 'sinusoid':
        return load_sinusoid(**kwargs)
    elif dataset_name == 'omniglot':
        return load_omniglot(**kwargs)
    elif dataset_name == 'miniimagenet':
        return load_miniimagenet(**kwargs)
    else:
        raise Exception("Dataset {0} not found".format(dataset_name))

def load_miniimagenet(num_classes=5):
    from data.miniimagenet import Miniimagenet
    data_dir = r"/data/ziz/not-backed-up/jxu/miniimagenet"
    train_set = Miniimagenet(data_dir, num_classes, 'train')
    val_set = Miniimagenet(data_dir, num_classes, 'val')
    return train_set, val_set

def load_omniglot(num_classes=5):
    import data.omniglot as og
    train_set, val_set = og.load_omniglot("/data/ziz/not-backed-up/jxu/omniglot")
    train_set = og.Omniglot(train_set, num_classes, dataset_name='omniglot-c{0}'.format(num_classes))
    val_set = og.Omniglot(val_set, num_classes, dataset_name='omniglot-c{0}'.format(num_classes))
    return train_set, val_set

def load_sinusoid(amp_range=[0.1, 5.0], phase_range=[0, np.pi], period_range=[2*np.pi, 2*np.pi], input_range=[-5., 5.]):
    from data.sinusoid import Sinusoid
    train_set = Sinusoid(amp_range, phase_range, period_range, input_range, dataset_name="sinusoid")
    val_set = train_set
    return train_set, val_set

def load_gpsamples():
    from data.gpsample import GPSample
    data = np.load("gpsamples_var05.npz")
    train_data = {"xs":data['xs'][:50000], "ys":data['ys'][:50000]}
    val_data = {"xs":data['xs'][50000:60000], "ys":data['ys'][50000:60000]}
    train_set = GPSample(input_range=[-2., 2.], var_range=[0.5, 0.5], max_num_samples=200, data=train_data)
    val_set = GPSample(input_range=[-2., 2.], var_range=[0.5, 0.5], max_num_samples=200, data=val_data)
    return train_set, val_set
