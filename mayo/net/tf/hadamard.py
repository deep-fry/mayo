import math

import scipy
import numpy as np
import tensorflow as tf


class HadamardLayers(object):
    @staticmethod
    def _is_power_of_two(num):
        return 2 ** int(math.log(num, 2)) == num

    def instantiate_zipf(self, node, tensor, params):
        channels = int(tensor.shape[-1])
        scales = np.array([1 / (n + 1) for n in range(channels)])
        scales = scales / np.sum(scales)
        return tensor * np.reshape(scales, [1, 1, 1, channels])

    def instantiate_hadamard(self, node, tensor, params):
        # check depth is 2^n
        channels = int(tensor.shape[-1])
        if 2 ** int(math.log(channels, 2)) != channels:
            raise ValueError(
                'Number of channels must be a power of 2 for hadamard layer.')

        # scale channels
        if params.get('variable_scales', False):
            # ensures correct broadcasting behaviour
            shape = (1, 1, 1, channels)
            scale = 1.0 / math.sqrt(channels)
            init = tf.truncated_normal_initializer(mean=scale, stddev=0.001)
            channel_scales = tf.get_variable(
                name='channel_scale', shape=shape, initializer=init)
            tensor *= channel_scales

        if params.pop('block', False):
            hadamard = scipy.linalg.hadamard(channels)
            hadamard = tf.constant(hadamard, dtype=tf.float32)
            # flatten input tensor
            flattened = tf.reshape(tensor, [-1, channels])
            # transform with hadamard
            transformed = flattened @ hadamard
            return tf.reshape(transformed, shape=tensor.shape)

        def fwht(value):
            if value.shape[-1] == 1:
                return value
            lower, upper = tf.split(value, 2, axis=-1)
            lower, upper = lower + upper, lower - upper
            return tf.concat((fwht(lower), fwht(upper)), axis=-1)

        # fast walsh-hadamard transform
        tensor = fwht(tensor)
        # normalization
        normalizer_fn = params.get('normalizer_fn', None)
        normalizer_params = params.get('normalizer_params', None)
        if normalizer_fn:
            tensor = normalizer_fn(tensor, **normalizer_params)
        # activation
        activation_fn = params.get('activation_fn', tf.nn.relu)
        if activation_fn is not None:
            tensor = activation_fn(tensor)
        return tensor

    def instantiate_zipf_hadamard_convolution(self, node, tensor, params):
        channels = int(tensor.shape[-1])
        out_channels = params.pop('num_outputs', channels)
        normalizer_fn = params.pop('normalizer_fn')
        normalizer_params = params.pop('normalizer_params')
        if normalizer_fn:
            params['biases_initializer'] = None
        activation_fn = params.pop('activation_fn', tf.nn.relu)
        block = params.pop('block', False)
        params['activation_fn'] = None
        if not self._is_power_of_two(channels):
            raise ValueError('Number of input channels must be 2^n.')
        if not self._is_power_of_two(out_channels):
            raise ValueError('Number of output channels must be 2^n.')
        if out_channels == channels:
            conv = self.instantiate_depthwise_convolution(node, tensor, params)
        elif out_channels > channels:
            conv_params = {
                'num_groups': channels,
                'num_outputs': out_channels / channels,
            }
            params.update(conv_params)
            conv = self.instantiate_convolution(node, tensor, params)
        zipf = self.instantiate_zipf(node, conv, None)
        hadamard_params = {
            'block': block,
            'normalizer_fn': normalizer_fn,
            'normalizer_params': normalizer_params,
            'activation_fn': activation_fn,
        }
        return self.instantiate_hadamard(node, zipf, hadamard_params)
