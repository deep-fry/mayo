import types
import functools
import contextlib

import tensorflow as tf


class ShapeError(ValueError):
    """Incorrect shape.  """


def map_fn(func, inputs, dtype=None, static=False):
    if not static:
        return tf.map_fn(func, inputs, dtype=dtype)
    inputs = [tf.unstack(i, axis=0) for i in inputs]
    outputs = []
    for args in zip(*inputs):
        outputs.append(func(args))
    return [tf.stack(o, axis=0) for o in zip(*outputs)]


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


def compose_functions(functions):
    def compose(f, g):
        return lambda x: g(f(x))
    return functools.reduce(compose, functions, lambda x: x)
