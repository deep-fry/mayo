import math
from functools import partial

import numpy as np
import tensorflow as tf


def _is_constant(*args):
    return all(isinstance(a, (bool, int, float)) for a in args)


def _is_numpy(*args):
    if _is_constant(*args):
        return False
    return all(isinstance(a, (bool, int, float, np.ndarray)) for a in args)


def _is_tensor(*args):
    return any(isinstance(a, tf.Tensor) for a in args)


def cast(value, dtype):
    if _is_constant(value):
        return dtype(value)
    if _is_numpy(value):
        dtypes = {
            float: np.float32,
            int: np.int32,
        }
        return np.cast[dtypes[dtype]](value)
    dtypes = {
        float: tf.float32,
        int: tf.int32,
    }
    return tf.cast(value, dtypes[dtype])


def where(bool_expr, true_value, false_value):
    if _is_constant(bool_expr, true_value, false_value):
        return true_value if bool_expr else false_value
    if _is_numpy(bool_expr, true_value, false_value):
        return np.where(bool_expr, true_value, false_value)
    return tf.where(bool_expr, true_value, false_value)


def sum(value):
    if _is_constant(value):
        raise TypeError
    if _is_numpy(value):
        return np.sum(value)
    return tf.reduce_sum(value)


def count(value):
    if _is_constant(value):
        raise TypeError
    if _is_numpy(value):
        return value.size
    return value.shape.num_elements()


def floor(value):
    if _is_constant(value):
        return math.floor(value)
    if _is_numpy(value):
        return np.floor(value)
    omap = {'Floor': 'Identity'}
    with tf.get_default_graph().gradient_override_map(omap):
        return tf.floor(value)


def round(value):
    if _is_constant(value):
        return round(value)
    if _is_numpy(value):
        return np.round(value)
    omap = {'Round': 'Identity'}
    with tf.get_default_graph().gradient_override_map(omap):
        return tf.round(value)


def abs(value):
    if _is_constant(value):
        return abs(value)
    if _is_numpy(value):
        return np.abs(value)
    return tf.abs(value)


def sqrt(value):
    if _is_constant(value):
        return math.sqrt(value)
    if _is_numpy(value):
        return np.sqrt(value)
    return tf.sqrt(value)


def log(value, base=None):
    if _is_constant(value, base):
        return math.log(value, base)
    if _is_numpy(value, base):
        return np.log(value) / np.log(base)
    return tf.log(value) / tf.log(cast(base, float))


def _binary_bool_operation(a, b, op):
    if _is_constant(a, b):
        raise TypeError('Element-wise operator not supported on scalars.')
    if _is_numpy(a, b):
        return getattr(np, op)(a, b)
    return getattr(tf, op)(a, b)


logical_or = partial(_binary_bool_operation, op='logical_or')
logical_and = partial(_binary_bool_operation, op='logical_and')


def _clip(*args, min_max=None):
    if _is_constant(*args):
        return min(*args) if min_max else max(*args)
    if _is_numpy(*args):
        return np.minimum(*args) if min_max else np.maximum(*args)
    return tf.minimum(*args) if min_max else tf.maximum(*args)


min = partial(_clip, min_max=True)
max = partial(_clip, min_max=False)


def binarize(tensor, threshold):
    return cast(abs(tensor) > threshold, float)


def clip_by_value(tensor, minimum, maximum, transparent_backprop=False):
    if not _is_tensor(tensor, minimum, maximum):
        return min(max(tensor, minimum), maximum)
    omap = {}
    if transparent_backprop:
        omap = {'Minimum': 'Identity', 'Maximum': 'Identity'}
    with tf.get_default_graph().gradient_override_map(omap):
        return tf.clip_by_value(tensor, minimum, maximum)
