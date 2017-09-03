import functools
import tensorflow as tf
import numpy as np

class Pruner(object):
    def __init__(self, net_params, masks):
        self.net_params = net_params
        self.masks = masks

    @staticmethod
    def threshold_based(x, params):
        if params['threshold']:
            threshold = params['threshold']
        else:
            raise ValueError('threshold based pruning requires a threshold value')
        mask = np.absolute(x) > threshold
        mask.astype(int)
        return mask

    @staticmethod
    def mean_std_based(x, params):
        if params['alpha']:
            alpha = params['alpha']
        else:
            raise ValueError('mean std based pruning requires an alpha value')
        threshold = np.mean(x) + alpha * np.std(x)
        mask = np.absolute(x) > threshold
        mask.astype(int)
        return mask


    def update_masks(self, prune_method, params):
        update_ops = []
        for key, item in self.net_params.items():
            func = getattr(self, prune_method)
            np_mask_value = func(item, params)
            update_op = tf.assign(self.masks[key+'_mask'], np_mask_value)
            update_ops.append(update_op)
        return tf.group(*update_ops)

class SurgeryFunctionCollection(object):
    """
    A collection of functions that can perform surgery on network.
    """
    @staticmethod
    def prune(value, name):
        shape = value.get_shape()
        mask_ini = tf.constant(np.ones(shape))
        mask = tf.get_variable(name + '_mask', initialier = mask_ini, trainable = False)
        value = value * mask
        return (value, mask)

    @staticmethod
    def rounder(value):
        omap = {'Round': 'Identity'}
        with tf.get_default_graph().gradient_override_map(omap):
            return tf.round(value)

    @classmethod
    def dynamic_fixed_point_quantizer(
            cls, value,
            integer_width=None, fractional_width=8, dynamic_range=0):
        """
        Quantize inputs into fixed-point values with 1-bit sign, an n-bit
        integer part and an f-bit fractional part with d-bit dynamic range.

        args:
            integer_width:
                the number of bits to use in integer part.  If not specified
                (None), then we do not restrict the value bound.
            fractional_width:
                the number of bits to use in fractional part.

        references:
            - https://arxiv.org/pdf/1604.03168
        """
        # x = f - d bits
        value *= 2 ** (fractional_width - dynamic_range)
        # quantize
        value = cls.rounder(value)
        # >> f
        value = tf.div(value, 2 ** fractional_width)
        # ensure number is representable without overflow
        if integer_width is not None:
            max_value = 2 ** (integer_width - dynamic_range)
            value = tf.clip_by_value(value, -max_value-1, max_value)
        # restore shift by dynamic range
        if dynamic_range != 0:
            value *= 2 ** dynamic_range
        return value

    @classmethod
    def fixed_point_quantizer(
            cls, value, integer_width=None, fractional_width=8):
        """
        Quantize inputs into fixed-point values with 1-bit sign, an n-bit
        integer part and an f-bit fractional part.
        """
        return cls.dynamic_fixed_point_quantizer(
            value, integer_width, fractional_width, 0)


def _create_object(func):
    def obj(**kwargs):
        return functools.partial(func, **kwargs)
    return obj


def _register_surgery_objects(module):
    for name in dir(SurgeryFunctionCollection):
        func = getattr(SurgeryFunctionCollection, name)
        if not callable(func) or name.startswith("__"):
            continue
        setattr(module, name, _create_object(func))
