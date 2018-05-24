import types
import functools
import contextlib

import tensorflow as tf


class ShapeError(ValueError):
    """Incorrect shape.  """


def debug(tensors):
    def wrapped(*args):
        __import__('ipdb').set_trace()
        return tf.ones([], dtype=tf.int32)
    original = tensors
    if isinstance(tensors, (tf.Tensor, tf.Variable)):
        tensors = [tensors]
    ones = tf.py_func(wrapped, tensors, [tf.int32])
    with tf.control_dependencies(ones):
        return original


def pad_to_shape(tensor, shape, default_value=0):
    # FIXME annoying hack for batching different sized shapes
    tensor_shape = tf.unstack(tf.shape(tensor))
    paddings = [
        [0, max_size - size]
        for max_size, size in zip(shape, tensor_shape)]
    tensor = tf.pad(tensor, paddings, constant_values=default_value)
    return tf.reshape(tensor, shape)


@contextlib.contextmanager
def null_scope():
    yield


def memoize_method(func):
    """
    A decorator to remember the result of the method call
    """
    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        name = '_memoize_{}'.format(func.__name__)
        try:
            return getattr(self, name)
        except AttributeError:
            result = func(self, *args, **kwargs)
            if isinstance(result, types.GeneratorType):
                # automatically resolve generators
                result = list(result)
            setattr(self, name, result)
            return result
    return wrapped


def memoize_property(func):
    return property(memoize_method(func))
