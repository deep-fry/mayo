from mayo.net.base import NetBase

import tensorflow as tf
from tensorflow.contrib import slim


class Net(NetBase):
    def instantiate_convolution(self, tensor, params):
        return slim.conv2d(tensor, **params)

    def instantiate_depthwise_separable_convolution(self, tensor, params):
        scope = params.pop('scope')
        num_outputs = params.pop('num_outputs', None)
        stride = params.pop('stride')
        kernel = params.pop('kernel_size')
        depth_multiplier = params.pop('depth_multiplier', 1)
        depthwise_regularizer = params.pop('depthwise_regularizer')
        if num_outputs is not None:
            pointwise_regularizer = params.pop('pointwise_regularizer')
        # depthwise layer
        depthwise = slim.separable_conv2d(
            tensor, num_outputs=None, kernel_size=kernel, stride=stride,
            weights_regularizer=depthwise_regularizer,
            depth_multiplier=1, scope='{}_depthwise'.format(scope), **params)
        if num_outputs is None:
            # if num_outputs is none, it is a depthwise by default
            return depthwise
        # pointwise layer
        num_outputs = max(int(num_outputs * depth_multiplier), 8)
        pointwise = slim.conv2d(
            depthwise, num_outputs=num_outputs, kernel_size=[1, 1], stride=1,
            weights_regularizer=pointwise_regularizer,
            scope='{}_pointwise'.format(scope), **params)
        return pointwise

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

    def instantiate_average_pool(self, tensor, params):
        self._reduce_kernel_size_for_small_input(params, tensor)
        return slim.avg_pool2d(tensor, **params)

    def instantiate_max_pool(self, tensor, params):
        self._reduce_kernel_size_for_small_input(params, tensor)
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
            tensor, **self._use_name_not_scope(params))

    def instantiate_squeeze(self, tensor, params):
        return tf.squeeze(tensor, **self._use_name_not_scope(params))

    def instantiate_flatten(self, tensor, params):
        return slim.flatten(tensor, **params)

    def instantiate_hadamard(self, tensor, params):
        import scipy
        # generate a hadmard matrix
        channels = int(tensor.shape[3])
        # spawn hadamard matrix from scipy
        hadamard_matrix = scipy.linalg.hadamard(channels)
        hadamard_matrix = tf.constant(hadamard_matrix, dtype=tf.float32)
        # large channel scales lead to divergence, hence rescale
        hadamard_matrix = hadamard_matrix / float(channels)
        tensor_reshaped = tf.reshape(tensor, [-1, channels])
        if params.get('scales', True):
            init = tf.truncated_normal_initializer(mean=1, stddev=0.001)
            channel_scales = tf.get_variable(
                name='channel_scale', shape=[channels], initializer=init)
            tensor_reshaped = tensor_reshaped * channel_scales
        transformed = tensor_reshaped @ hadamard_matrix
        transformed = tf.reshape(transformed, shape=transformed.shape)
        transformed = tf.concat(transformed, 3)
        activation_function = params.get('activation_fn', tf.nn.relu)
        if activation_function is not None:
            transformed = activation_function(transformed)
        return transformed

    def instantiate_concat(self, tensors, params):
        return [tf.concat(tensors, **self._use_name_not_scope(params))]
