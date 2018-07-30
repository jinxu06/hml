import numpy as np


def load(dataset_name, **kwargs):
    if dataset_name == 'gpsamples':
        return load_gpsamples(**kwards)
    elif dataset_name == 'sinusoid':
        return load_sinusoid(**kwards)
    elif dataset_name == 'omniglot':
        return load_omniglot(**kwargs)
    else:
        raise Exception("Dataset {0} not found".format(dataset_name))


def load_omniglot():
    pass

def load_sinusoid():
    from data.sinusoid import Sinusoid
    train_set = Sinusoid(amp_range=[0.1, 5.0], phase_range=[0, np.pi], period_range=[2*np.pi, 2*np.pi], input_range=[-5., 5.], dataset_name=args.dataset_name)
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
