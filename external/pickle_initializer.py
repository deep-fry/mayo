import pickle

import tensorflow as tf


_init_params = {}


def _params(path):
    params = _init_params.get(path, None)
    if params is None:
        with open(path, 'rb') as f:
            params = pickle.load(f)
        _init_params[path] = params
    return params


def initializer(path, net, name, base):
    name = '{}/{}/{}'.format(net, name, base)
    params = _params(path)
    return tf.constant_initializer(params[name])

def vgg_initializer(path, net, name, base):
    # name = '{}/{}/{}'.format(net, layer_name, base)
    params = _params(path)
    for key in params.keys():
        if (base in key) and (name in key):
            name = key
            break
    return tf.constant_initializer(params[name])
