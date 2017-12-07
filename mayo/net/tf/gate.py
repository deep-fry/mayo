import math

import tensorflow as tf

from mayo.util import Percent


def _sparsity(session, collection):
    # sparsity info
    gates = tf.get_collection(collection)
    if not gates:
        return
    valid = []
    total = 0
    for g in gates:
        valid.append(tf.reduce_sum(tf.cast(g, tf.float32)))
        total += g.shape.num_elements()
    density = tf.add_n(valid) / total
    density_formatter = lambda d: Percent(
        session.change.moving_metrics('density', d, std=False))
    session.register_update('density', density, density_formatter)


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
        constructor, conv_input, conv_output, density,
        activation_fn, online, scope):
    """
    Regularize gate by making gate output to predict whether subsampled
    conv output is in top-`density` elements as close as possible.

    conv_input (tf.Tensor): The input of the convolution layer.
    conv_output (tf.Tensor): The output from convolution layer.
    density (float): The target density.
    activation_fn: The activation function used.
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
    loss = tf.losses.mean_squared_error(
        match, gate_output, weights=0.01,
        loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)

    # gating loss for printing
    def update_func(session, collection):
        def formatter(loss):
            loss_mean, loss_std = session.change.moving_metrics(
                'gate.loss', loss)
            loss_std = '±{}'.format(Percent(loss_std / loss_mean))
            return '{:10f}{:5}'.format(loss_mean, loss_std)
        loss = tf.get_collection(collection)[0]
        session.register_update('gate loss', loss, formatter)
    constructor.register_update('mayo.gate.loss', loss, update_func)

    if online:
        return _descriminate_by_density(gate_output, density)
    return gate_output > 0


class GateLayers(object):
    """
    Layer implementations for gated convolution.
    """
    def instantiate_gated_convolution(self, node, tensor, params):
        online = params.pop('online', False)
        density = params.pop('density')
        activation_fn = params.get('activation_fn', tf.nn.relu)
        # convolution
        output = self.instantiate_convolution(None, tensor, params)
        # predictor policy
        gate = _regularized_gate(
            self, tensor, output, density, activation_fn, online,
            params['scope'])
        # register gate sparsity for printing
        self.register_update('mayo.gates', gate, _sparsity)
        # actual gating
        gate = tf.stop_gradient(tf.cast(gate, tf.float32))
        return output * gate

    def instantiate_gate(self, node, tensor, params):
        gate_scope = '{}/gate'.format(params['scope'])
        subsampled = _subsample(self, tensor, gate_scope)
        gate = _descriminate_by_density(subsampled, params['density'])
        # register gate sparsity for printing
        self.register_update('mayo.gates', gate, _sparsity)
        # actual gating
        gate = tf.stop_gradient(tf.cast(gate, tf.float32))
        return tensor * gate
