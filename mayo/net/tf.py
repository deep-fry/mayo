import math

import tensorflow as tf
from tensorflow.contrib import slim

from mayo.util import object_from_params
from mayo.net.util import use_name_not_scope
from mayo.net.tfbase import TFNetBase


class TFNet(TFNetBase):
    """ Create a TensorFlow graph from "config.model" model definition.  """

    def instantiate_convolution(self, tensor, params):
        return slim.conv2d(tensor, **params)

    def instantiate_depthwise_separable_convolution(self, tensor, params):
        scope = params.pop('scope')
        num_outputs = params.pop('num_outputs', None)
        # depthwise layer
        stride = params.pop('stride')
        kernel = params.pop('kernel_size')
        depth_multiplier = params.pop('depth_multiplier', 1)
        depthwise_regularizer = params.pop('depthwise_regularizer')
        # pop it out, so later **params is correct
        pointwise_regularizer = params.pop('pointwise_regularizer', None)
        depthwise = slim.separable_conv2d(
            tensor, num_outputs=None, kernel_size=kernel, stride=stride,
            weights_regularizer=depthwise_regularizer, depth_multiplier=1,
            scope='{}/depthwise'.format(scope), **params)
        if num_outputs is None:
            # skip pointwise if `num_outputs` is not specified
            return depthwise
        # pointwise layer
        num_outputs = max(int(num_outputs * depth_multiplier), 8)
        pointwise = slim.conv2d(
            depthwise, num_outputs=num_outputs, kernel_size=[1, 1], stride=1,
            weights_regularizer=pointwise_regularizer,
            scope='{}/pointwise'.format(scope), **params)
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
            tensor, **use_name_not_scope(params))

    def instantiate_squeeze(self, tensor, params):
        return tf.squeeze(tensor, **use_name_not_scope(params))

    def instantiate_flatten(self, tensor, params):
        return slim.flatten(tensor, **params)

    def instantiate_local_gating(self, tensor, params):
        num, height, width, channels = tensor.shape
        policy = params.pop('policy')
        gate_scope = '{}/gate'.format(params['scope'])
        pool_params = {
            'padding': 'VALID',
            'kernel_size': [height, width],
            'scope': gate_scope,
        }
        # max pool is hardware-friendlier
        gate = self.instantiate_max_pool(tensor, pool_params)
        # fc
        fc_params = {
            'kernel_size': 1,
            'num_outputs': channels,
            'biases_initializer': tf.ones_initializer(),
            'weights_initializer':
                tf.truncated_normal_initializer(stddev=0.01),
            'activation_fn': None,
            'scope': gate_scope,
        }
        gate = self.instantiate_convolution(gate, fc_params)
        # regularizer policy
        regu_cls, regu_params = object_from_params(policy)
        regularization = regu_cls(**regu_params)(gate)
        tf.add_to_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES, regularization)
        # threshold
        omap = {'Sign': 'Identity'}
        with tf.get_default_graph().gradient_override_map(omap):
            gate = tf.sign(gate)
            gate = tf.clip_by_value(gate, 0, 1)
        tf.add_to_collection('mayo.gates', gate)
        # gating
        return tensor * gate

    def instantiate_gated_convolution(self, tensor, params):
        num, height, width, channels = tensor.shape
        policy = params.pop('policy')
        # convolution
        output = self.instantiate_convolution(tensor, params)
        # gating network
        gate_scope = '{}/gate'.format(params['scope'])
        # pool
        pool_params = {
            'padding': 'VALID',
            'kernel_size': [height, width],
            'scope': gate_scope,
        }
        # max pool is hardware-friendlier
        gate = self.instantiate_max_pool(tensor, pool_params)
        # fc
        num_outputs = params['num_outputs']
        fc_params = {
            'kernel_size': 1,
            'num_outputs': num_outputs,
            'biases_initializer': tf.ones_initializer(),
            'weights_initializer':
                tf.truncated_normal_initializer(stddev=0.01),
            'activation_fn': None,
            'scope': gate_scope,
        }
        gate = self.instantiate_convolution(gate, fc_params)
        # policies
        if policy.type == 'softmax_cross_entropy':
            # predictor policy
            # output pool
            _, out_height, out_width, out_channels = output.shape
            pool_params = {
                'padding': 'VALID',
                'kernel_size': [out_height, out_width],
                'scope': gate_scope,
            }
            # TODO is it really sensible to use averge pool as the
            # threshold criteria?
            avg_output = self.instantiate_max_pool(output, pool_params)
            # not training the output as we train the predictor `gate`
            avg_output = tf.stop_gradient(avg_output)
            # loss
            tf.losses.softmax_cross_entropy(
                avg_output, gate, weights=0.00001,
                loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
            # thresholding
            k = math.ceil(int(out_channels) * policy.density)
            gate_top_k, _ = tf.nn.top_k(gate, k=k, sorted=True)
            gate_threshold = tf.reduce_min(gate_top_k, axis=[1, 2, 3])
            # gate_max.shape [batch_size]
            for _ in range(3):
                gate_threshold = tf.expand_dims(gate_threshold, -1)
            # gate_max.shape [batch_size, 1, 1, 1]
            gate = tf.cast(gate > gate_threshold, tf.float32)
            # training happens in softmax_cross_entropy
            gate = tf.stop_gradient(gate)
        else:
            # regularizer policy
            regu_cls, regu_params = object_from_params(policy)
            regularization = regu_cls(**regu_params)(gate)
            tf.add_to_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES, regularization)
            # threshold
            omap = {'Sign': 'Identity'}
            with tf.get_default_graph().gradient_override_map(omap):
                gate = tf.sign(gate)
                gate = tf.clip_by_value(gate, 0, 1)
        tf.add_to_collection('mayo.gates', gate)
        # gating
        return output * gate

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
        # hadamard transform
        tensor = fwht(tensor)
        # activation
        activation_function = params.get('activation_fn', tf.nn.relu)
        if activation_function is not None:
            tensor = activation_function(tensor)
        return tensor

    def instantiate_concat(self, tensors, params):
        return tf.concat(tensors, **use_name_not_scope(params))
