import math

import tensorflow as tf


class HadamardLayers(object):
    def instantiate_hadamard(self, tensor, params):
        def fwht(value):
            if value.shape[-1] == 1:
                return value
            lower, upper = tf.split(value, 2, axis=-1)
            lower, upper = lower + upper, lower - upper
            return tf.concat((fwht(lower), fwht(upper)), axis=-1)
        # check depth is 2^n
        channels = int(tensor.shape[-1])
        if 2 ** int(math.log(channels, 2)) != channels:
            raise ValueError(
                'Number of channels must be a power of 2 for hadamard layer.')
        # scale channels
        scale = 1.0 / math.sqrt(channels)
        if params.get('variable_scales', False):
            # ensures correct broadcasting behaviour
            shape = (1, 1, 1, channels)
            init = tf.truncated_normal_initializer(mean=scale, stddev=0.001)
            channel_scales = tf.get_variable(
                name='channel_scale', shape=shape, initializer=init)
            tensor *= channel_scales
        else:
            tensor *= scale
        # hadamard transform
        tensor = fwht(tensor)
        # activation
        activation_function = params.get('activation_fn', tf.nn.relu)
        if activation_function is not None:
            tensor = activation_function(tensor)
        return tensor
