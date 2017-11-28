import math

import tensorflow as tf

from mayo.util import Percent


class GateLayers(object):
    def _gate_sparsity(self, session, collection):
        # sparsity info
        gates = tf.get_collection(collection)
        if not gates:
            return
        valid = tf.add_n([tf.reduce_sum(g) for g in gates])
        total = sum(g.shape.num_elements() for g in gates)
        density = valid / total
        density_formatter = lambda d: Percent(
            session.change.moving_metrics('density', d, std=False))
        session.register_update('density', density, density_formatter)

    def _gate_subsample(self, tensor, scope):
        num, height, width, channels = tensor.shape
        # pool
        pool_params = {
            'padding': 'VALID',
            'kernel_size': [height, width],
            'scope': scope,
        }
        gate_input = tf.stop_gradient(tensor)
        # max pool is hardware-friendlier
        return self.instantiate_max_pool(gate_input, pool_params)

    def _gate_network(self, tensor, params, scope):
        gate_input = self._gate_subsample(tensor, scope)
        # fc
        fc_params = {
            'kernel_size': 1,
            'num_outputs': params['num_outputs'],
            'biases_initializer': tf.zeros_initializer(),
            'weights_initializer':
                tf.truncated_normal_initializer(stddev=0.01),
            'activation_fn': params.get('activation_fn', tf.nn.relu),
            'scope': scope,
        }
        return self.instantiate_convolution(gate_input, fc_params)

    def _gate_regularizer(self, conv_output, gate_output, density):
        """
        Regularize gate by making gate output to predict whether subsampled
        conv output is in top-`density` elements as close as possible.
        """
        # output pool
        _, out_height, out_width, out_channels = conv_output.shape
        pool_params = {
            'padding': 'VALID',
            'kernel_size': [out_height, out_width],
        }
        subsampled = self.instantiate_max_pool(conv_output, pool_params)
        # not training with the output as we train the predictor `gate`
        subsampled = tf.stop_gradient(subsampled)
        num_active = math.ceil(int(out_channels) * density)
        subsampled_top, _ = tf.nn.top_k(subsampled, k=num_active)
        subsampled_threshold = tf.reduce_min(subsampled_top, axis=[1, 2, 3])
        # [batch_size]
        for _ in range(3):
            subsampled_threshold = tf.expand_dims(subsampled_threshold, -1)
        # [batch_size, 1, 1, 1]
        descriminator = subsampled > subsampled_threshold
        ones = tf.ones(descriminator.shape)
        descriminator = tf.where(descriminator, ones, -ones)
        # loss to minimize
        loss = tf.losses.mean_squared_error(
            descriminator, gate_output, weights=0.01,
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
        self.register_update('mayo.gate.loss', loss, update_func)

    def instantiate_gated_convolution(self, node, tensor, params):
        density = params.pop('density')
        # gating network
        gate_scope = '{}/gate'.format(params['scope'])
        gate = self._gate_network(tensor, params, gate_scope)
        # convolution
        output = self.instantiate_convolution(tensor, params)
        # predictor policy
        self._gate_regularizer(output, gate, density)
        # thresholding
        gate = tf.cast(gate > 0, tf.float32)
        # training happens in regularizer
        gate = tf.stop_gradient(gate)
        # register gate sparsity for printing
        self.register_update('mayo.gates', gate, self._gate_sparsity)
        # actual gating
        return output * gate

    def instantiate_gate(self, node, tensor, params):
        gate_scope = '{}/gate'.format(params['scope'])
        subsample = self._gate_subsample(tensor, gate_scope)
        # TODO make threshold a variable
        gate = tf.cast(subsample > params['threshold'], tf.float32)
        self.register_update('mayo.gates', gate, self._gate_sparsity)
        # gating
        return tensor * tf.stop_gradient(gate)
