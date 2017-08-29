import pickle

import yaml
import numpy as np
import tensorflow as tf


_init_params = {}
_formatter = {}


def _params(path):
    params = _init_params.get(path, None)
    if params is None:
        with open(path, 'rb') as f:
            params = pickle.load(f)
        _init_params[path] = params
    return params


def formatter(file):
    f = _formatter.get(file, None)
    if f is None:
        with open(file, 'r') as file:
            f = yaml.load(file)
        _formatter[file] = f
    return f


def initializer(path, net, name, base, perturb=0.0):
    #  name = '{}/{}/{}'.format(net, name, base)
    # FIXME grep-based hack
    params = _params(path)
    for key in params.keys():
        if (base in key) and (name in key):
            name = key
            break
    tensor = params[name]
    if perturb > 0:
        std = np.std(tensor)
        tensor += np.random.normal(0, perturb * std, tensor.shape)
    return tf.constant_initializer(tensor, verify_shape=True)
