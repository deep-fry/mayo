import math

import numpy as np
import tensorflow as tf

from mayo.util import Percent, memoize_method


def _subsample(constructor, tensor, scope):
    num, height, width, channels = tensor.shape
    # pool
    pool_params = {
        'padding': 'VALID',
        'kernel_size': [height, width],
        'scope': scope,
    }
    gate_input = tf.stop_gradient(tensor)
    # max pool is hardware-friendlier
    return constructor.instantiate_max_pool(None, gate_input, pool_params)


def _gate_network(constructor, tensor, num_outputs, activation_fn, scope):
    gate_input = _subsample(constructor, tensor, scope)
    # fc
    fc_params = {
        'kernel_size': 1,
        'num_outputs': num_outputs,
        'biases_initializer': tf.zeros_initializer(),
        'weights_initializer': tf.truncated_normal_initializer(stddev=0.01),
        'activation_fn': activation_fn,
        'scope': scope,
    }
    return constructor.instantiate_convolution(None, gate_input, fc_params)


def _descriminate_by_density(to_gate, density):
    if not (0 < density <= 1):
        raise ValueError(
            'Gate density value {} is out of range (0, 1].'.format(density))
    # not training with the output as we train the predictor `gate`
    to_gate = tf.stop_gradient(to_gate)
    # number of active channels
    channels = to_gate.shape[-1]
    num_active = math.ceil(int(channels) * density)
    # top_k, where k is the number of active channels
    top, _ = tf.nn.top_k(to_gate, k=num_active)
    # disable channels with smaller activations
    threshold = tf.reduce_min(top, axis=[1, 2, 3], keep_dims=True)
    return tf.stop_gradient(to_gate >= threshold)


def _regularized_gate(
        constructor, node, conv_input, conv_output, density,
        activation_fn, online, weight, scope):
    """
    Regularize gate by making gate output to predict whether subsampled
    conv output is in top-`density` elements as close as possible.

    node (mayo.net.graph.LayerNode): The convolution layer node.
    conv_input (tf.Tensor): The input of the convolution layer.
    conv_output (tf.Tensor): The output from convolution layer.
    density (float): The target density.
    activation_fn: The activation function used.
    weight (float): The weight of the gate regularizer loss.
    online (bool): The switch to compute top_k online or offline.

    Returns the regularized gate (1: enable, 0: disable).
    """
    # gating network
    num_outputs = conv_output.shape[-1]
    gate_scope = '{}/gate'.format(scope)
    if not online:
        activation_fn = None
    gate_output = _gate_network(
        constructor, conv_input, num_outputs, activation_fn, gate_scope)
    # output pool
    _, out_height, out_width, out_channels = conv_output.shape
    pool_params = {
        'padding': 'VALID',
        'kernel_size': [out_height, out_width],
    }
    subsampled = constructor.instantiate_max_pool(
        None, conv_output, pool_params)

    # training
    if online:
        # online descriminator: we simply match max values in each channel
        match = subsampled
    else:
        # offline descriminator: we train the gate to produce 1 for active
        # channels and -1 for gated channels
        match = _descriminate_by_density(subsampled, density)
        ones = tf.ones(match.shape)
        match = tf.where(match, ones, -ones)

    # loss regularizer
    loss = tf.losses.mean_squared_error(
        match, gate_output, weights=weight,
        loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
    constructor.estimator.register(loss, 'loss', node)

    if online:
        return _descriminate_by_density(gate_output, density)
    return gate_output > 0


class GateLayers(object):
    """Layer implementations for gated convolution.  """

    def _register_gate_density(self, node, gate, in_channels):
        history = None if self.is_training else 'infinite'
        self.estimator.register(gate, 'gate', node, history=history)

    @staticmethod
    def _gate_loss_formatter(estimator):
        # gating loss for printing
        losses = estimator.get_histories('loss')
        total_losses = None
        for loss_history in losses.values():
            if total_losses is None:
                total_losses = list(loss_history)
            else:
                total_losses = [
                    a + b for a, b in zip(total_losses, loss_history)]
        loss_mean = np.mean(total_losses)
        loss_std = Percent(np.std(total_losses) / loss_mean)
        return 'gate.loss: {:.5f}±{}'.format(loss_mean, loss_std)

    @staticmethod
    def _gate_density_formatter(estimator):
        gates = estimator.get_values('gate')
        valid = total = 0
        for layer, gate in gates.items():
            valid += np.sum(gate.astype(np.float32))
            total += gate.size
        return 'gate: {}'.format(Percent(valid / total))

    @memoize_method
    def _register_gate_formatters(self):
        self.estimator.register_formatter(self._gate_loss_formatter)
        self.estimator.register_formatter(self._gate_density_formatter)

    def instantiate_gated_convolution(self, node, tensor, params):
        density = params.pop('density')
        online = params.pop('online', False)
        should_gate = params.pop('should_gate', True)
        weight = params.pop('weight', 0.01)
        activation_fn = params.get('activation_fn', tf.nn.relu)
        # convolution
        output = self.instantiate_convolution(None, tensor, params)
        # predictor policy
        gate = _regularized_gate(
            self, node, tensor, output, density, activation_fn, online,
            weight, params['scope'])
        # register gate sparsity for printing
        self._register_gate_density(node, gate, tensor.shape[-1])
        self._register_gate_formatters()
        if not should_gate:
            return output
        # actual gating
        gate = tf.stop_gradient(tf.cast(gate, tf.float32))
        return output * gate

    def instantiate_gate(self, node, tensor, params):
        gate_scope = '{}/gate'.format(params['scope'])
        subsampled = _subsample(self, tensor, gate_scope)
        gate = _descriminate_by_density(subsampled, params['density'])
        # register gate sparsity for printing
        self._register_gate_density(node, gate, tensor.shape[-1])
        self._register_gate_formatters()
        # actual gating
        gate = tf.stop_gradient(tf.cast(gate, tf.float32))
        if not params['should_gate']:
            with tf.control_dependencies([gate]):
                return tensor
        return tensor * gate
