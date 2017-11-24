import math

import tensorflow as tf
from tensorflow.contrib import slim

from mayo.net.tf.util import use_name_not_scope
from mayo.net.tf.base import TFNetBase


class Layers(TFNetBase):
    """ Create a TensorFlow graph from "config.model" model definition.  """

    def instantiate_convolution(self, tensor, params):
        return slim.conv2d(tensor, **params)

    def instantiate_depthwise_convolution(self, tensor, params):
        multiplier = params.pop('depth_multiplier', 1)
        return slim.separable_conv2d(
            tensor, num_outputs=None, depth_multiplier=multiplier, **params)

    @staticmethod
    def _reduce_kernel_size_for_small_input(params, tensor):
        shape = tensor.get_shape().as_list()
        if shape[1] is None or shape[2] is None:
            return
        kernel = params['kernel_size']
        if isinstance(kernel, int):
            kernel = [kernel, kernel]
        stride = params.get('stride', 1)
        params['kernel_size'] = [
            min(shape[1], kernel[0]), min(shape[2], kernel[1])]
        # tensorflow complains when stride > kernel size
        params['stride'] = min(stride, kernel[0], kernel[1])

    def _should_pool_nothing(self, params):
        # skip pooling with 1x1 kernel @ stride 1, which is a no-op
        return params['kernel_size'] in (1, [1, 1]) and params['stride'] == 1

    def instantiate_average_pool(self, tensor, params):
        self._reduce_kernel_size_for_small_input(params, tensor)
        if self._should_pool_nothing(params):
            return tensor
        return slim.avg_pool2d(tensor, **params)

    def instantiate_max_pool(self, tensor, params):
        self._reduce_kernel_size_for_small_input(params, tensor)
        if self._should_pool_nothing(params):
            return tensor
        return slim.max_pool2d(tensor, **params)

    def instantiate_fully_connected(self, tensor, params):
        return slim.fully_connected(tensor, **params)

    def instantiate_softmax(self, tensor, params):
        return slim.softmax(tensor, **params)

    def instantiate_dropout(self, tensor, params):
        params['is_training'] = self.is_training
        return slim.dropout(tensor, **params)

    def instantiate_local_response_normalization(self, tensor, params):
        return tf.nn.local_response_normalization(
            tensor, **use_name_not_scope(params))

    def instantiate_squeeze(self, tensor, params):
        return tf.squeeze(tensor, **use_name_not_scope(params))

    def instantiate_flatten(self, tensor, params):
        return slim.flatten(tensor, **params)

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

    def instantiate_concat(self, tensors, params):
        return tf.concat(tensors, **use_name_not_scope(params))
