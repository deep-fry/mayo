import functools
import tensorflow as tf


class SurgeryFunctionCollection(object):
    """
    A collection of functions that can perform surgery on network.
    """
    @staticmethod
    def pruner(value, mask):
        return value * mask

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
        Quantize inputs into 2's compliment fixed-point values with an n-bit
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
            value = tf.clip_by_value(value, -max_value, max_value - 1)
        # restore shift by dynamic range
        if dynamic_range != 0:
            value *= 2 ** dynamic_range
        return value

    @classmethod
    def fixed_point_quantizer(
            cls, value, integer_width=None, fractional_width=8):
        """
        Quantize inputs into 2's compliment fixed-point values with an n-bit
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
