import math

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from mayo.util import Percent, memoize_method


class GateError(Exception):
    """Gating-related exceptions.  """


class GateParameterValueError(GateError):
    """Incorrect parameters used.  """


class GateGranularityTypeError(GateError):
    """Incorrect granularity used.  """


class GatePolicyTypeError(GateError):
    """Unrecognized policy.  """


class GatedConvolutionInstantiator(object):
    _accepted_policies = ['naive', 'parametric_gamma', 'squeeze_excitation']
    _accepted_granularities = ['channel', 'vector']
    _normalizer_names = {
        slim.batch_norm: 'BatchNorm',
    }

    def __init__(
            self, constructor, node, conv_params, gate_params, conv_input):
        super().__init__()
        self.constructor = constructor
        self.is_training = constructor.is_training
        self.node = node

        self.kernel_size = conv_params['kernel_size']
        self.stride = conv_params.get('stride', 1)
        self.padding = conv_params.get('padding', 'SAME')
        self.num_outputs = conv_params['num_outputs']
        self.scope = conv_params['scope']
        # delay normalization
        self.normalizer_fn = conv_params.pop('normalizer_fn', None)
        self.normalizer_params = conv_params.pop('normalizer_params', None)
        if self.normalizer_fn:
            # disable bias
            conv_params['biases_initializer'] = None
        # delay activation
        self.activation_fn = conv_params.get('activation_fn', tf.nn.relu)
        conv_params['activation_fn'] = None

        # gate params
        self.policy = gate_params['policy']
        self.density = gate_params['density']
        self.granularity = gate_params['granularity']
        self.pool = gate_params['pool']
        self.weight = gate_params['weight']
        self.squeeze_factor = gate_params['squeeze_factor']
        self.should_gate = gate_params['should_gate']
        self._check_policy()
        self._check_granularity()

        # convolution input & output
        self.input = conv_input
        self.output = constructor.instantiate_convolution(
            node, conv_input, conv_params)

    def _check_policy(self):
        if self.policy not in self._accepted_policies:
            raise GatePolicyTypeError(
                'Unrecognized policy {}, we accept one of {}.'
                .format(self.policy, ', '.join(self._accepted_policies)))
        if self.policy == 'parametric_gamma':
            if self.normalizer_fn is not slim.batch_norm:
                raise GatePolicyTypeError(
                    'Policy "{}" is used, we expect slim.batch_norm to '
                    'be used but it is absent in {}.'
                    .format(self.policy, self.node))

    def _check_granularity(self):
        if self.granularity in self._accepted_granularities:
            return
        raise GateGranularityTypeError(
            'Unrecognized granularity {}, we accept one of {}.'
            .format(self.granularity, ', '.join(self._accepted_granularities)))

    def subsample(self, tensor):
        num, height, width, channels = tensor.shape
        if self.granularity == 'channel':
            kernel = [height, width]
        elif self.granularity == 'vector':
            kernel = [1, width]
        else:
            self._check_granularity()

        # pool
        pool_params = {
            'padding': 'VALID',
            'kernel_size': kernel,
            'stride': 1,
            'scope': '{}/subsample'.format(self.scope),
        }
        if self.pool == 'max':
            # max pool is hardware-friendlier
            subsampled = self.constructor.instantiate_max_pool(
                None, tensor, pool_params)
        elif self.pool == 'l2':
            # FIXME this cannot do vector-wise
            subsampled = tf.nn.l2_loss(tensor)
            # tensor = tf.square(tensor)
            # subsampled = constructor.instantiate_average_pool(
            #     None, tensor, pool_params)
        elif self.pool in ('l1', 'avg'):
            if self.pool == 'l1':
                tensor = tf.abs(tensor)
            subsampled = self.constructor.instantiate_average_pool(
                None, tensor, pool_params)
        else:
            raise GateParameterValueError(
                'feature extract type not supported.')
        num, height, width, channels = subsampled.shape

        if self.granularity == 'channel' and not (height == width == 1):
            raise GateParameterValueError(
                'We expect subsampled image for channel granularity '
                'to be 1x1.')
        if self.granularity == 'vector' and width != 1:
            raise GateParameterValueError(
                'We expect subsampled width for vector granularity to be 1.')
        return subsampled

    @memoize_method
    def subsampled_output(self):
        return self.subsample(self.output)

    def _regularize(self, gate):
        """
        Regularize gate by making gate output to predict whether subsampled
        conv output is in top-`density` elements as close as possible.
        """
        if self.weight <= 0:
            return
        loss = None
        loss_name = tf.GraphKeys.REGULARIZATION_LOSSES
        if self.policy == 'naive':
            match = tf.stop_gradient(self.subsampled_output())
        elif self.policy == 'parametric_gamma':
            match = None
        elif self.policy == 'squeeze_excitation':
            match = tf.stop_gradient(self.squeeze_excitation())
        else:
            self._check_policy()
        if match is not None:
            # training
            # policy descriminator: we simply match values in each channel
            # using a loss regularizer
            loss = tf.losses.mean_squared_error(
                match, gate, loss_collection=None)
        else:
            # parametric gamma does not match anything
            loss = tf.nn.l2_loss(gate)
        loss *= self.weight
        tf.add_to_collection(loss_name, loss)
        self.constructor.session.estimator.register(
            loss, 'gate.loss', self.node)

    @memoize_method
    def gate(self):
        subsampled = self.subsample(self.input)
        if self.granularity == 'channel':
            kernel = 1
            stride = 1
            padding = 'VALID'
        elif self.granularity == 'vector':
            if isinstance(self.kernel_size, int):
                kernel_height = self.kernel_size
            else:
                kernel_height, _ = self.kernel_size
            kernel = [kernel_height, 1]
            if not isinstance(self.padding, str):
                if isinstance(self.padding, int):
                    padding_height = self.padding
                else:
                    padding_height, _ = self.padding
                padding = [self.padding, 0]
            if isinstance(self.stride, int):
                stride_height = self.stride
            else:
                stride_height, _ = self.stride
            stride = [stride_height, 1]
        else:
            self._check_granularity()
        normalizer_fn = None if self.policy == 'naive' else self.normalizer_fn
        params = {
            'kernel_size': kernel,
            'stride': stride,
            'padding': padding,
            'num_outputs': self.num_outputs,
            'biases_initializer': tf.constant_initializer(1.0),
            'weights_initializer':
                tf.truncated_normal_initializer(stddev=0.01),
            # FIXME should we use normalizer_fn?
            'normalizer_fn': normalizer_fn,
            'normalizer_params': self.normalizer_params,
            'activation_fn': self.activation_fn,
            'scope': '{}/gate'.format(self.scope),
        }
        padded = self.constructor.instantiate_numeric_padding(
            None, subsampled, params)
        output = self.constructor.instantiate_convolution(None, padded, params)
        self._regularize(output)
        return output

    @memoize_method
    def actives(self):
        """
        Mark a portion of top elements in gate output to true,
        where the portion is approximately the specified density.
        """
        if not (0 < self.density <= 1):
            raise GateParameterValueError(
                'Gate density value {} is out of range (0, 1].'
                .format(self.density))
        # not training with the output as we train the predictor `gate`
        tensor = tf.stop_gradient(self.gate())
        # number of active elemetns
        num, height, width, channels = tensor.shape
        if self.granularity == 'channel':
            num_elements = channels
        elif self.granularity == 'vector':
            num_elements = width * channels
        else:
            self._check_granularity()
        num_active = math.ceil(int(num_elements) * self.density)
        if num_active == num_elements:
            # all active, not gating
            return tensor
        # reshape the last dimensions into one
        reshaped = tf.reshape(tensor, [num, -1])
        # top_k, where k is the number of active channels
        top, _ = tf.nn.top_k(reshaped, k=(num_active + 1))
        # disable channels with smaller activations
        threshold = tf.reduce_min(top, axis=[1], keep_dims=True)
        active = tf.reshape(reshaped > threshold, tensor.shape)
        return tf.stop_gradient(active)

    @memoize_method
    def squeeze_excitation(self):
        def params(name, outputs):
            scope = '{}/SqueezeExcitation/{}'.format(self.scope, name)
            return {
                'kernel_size': 1,
                'stride': 1,
                'num_outputs': outputs,
                'scope': scope,
                'weights_initializer':
                    tf.truncated_normal_initializer(stddev=0.01),
                'biases_initializer': tf.constant_initializer(value=1.0),
            }
        num_squeezed = math.ceil(self.num_outputs / float(self.squeeze_factor))
        conv = self.constructor.instantiate_convolution
        squeeze_params = params('Squeeze', num_squeezed)
        squeezed = conv(None, self.subsampled_output(), squeeze_params)
        expand_params = params('Expand', self.num_outputs)
        return conv(None, squeezed, expand_params)

    def _instantiate_normalizer(self, prenorm):
        # normalization
        if not self.normalizer_fn:
            return prenorm
        if self.policy == 'parametric_gamma':
            normalizer_params = dict(self.normalizer_params, **{
                'scale': False,
                'center': False,
                'activation_fn': None,
                'scope': '{}/BatchNorm'.format(self.scope),
                'is_training': self.is_training,
            })
            output = self.constructor.instantiate_batch_normalization(
                None, self.output, normalizer_params)
            beta_scope = '{}/gate/shift'.format(self.scope)
            beta = tf.get_variable(
                beta_scope, shape=output.shape[-1], dtype=tf.float32,
                initializer=tf.constant_initializer(0.1), trainable=True)
            # gate output is the parametric gamma value
            return self.gate() * output + beta
        scope = '{}/{}'.format(
            self.scope, self._normalizer_names[self.normalizer_fn])
        normalizer_params = dict(self.normalizer_params, scope=scope)
        return self.normalizer_fn(self.output, **normalizer_params)

    def instantiate(self):
        output = self._instantiate_normalizer(self.output)
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        # gating
        if self.should_gate:
            output *= tf.cast(self.actives(), tf.float32)
        self._register()
        return output

    def _register(self):
        history = None if self.is_training else 'infinite'
        self.constructor.session.estimator.register(
            self.gate(), 'gate.output', self.node, history=history)
        if self.should_gate:
            self.constructor.session.estimator.register(
                self.actives(), 'gate.active', self.node, history=history)


class GateLayers(object):
    """Layer implementations for gated convolution.  """

    @staticmethod
    def _gate_loss_formatter(estimator):
        # gating loss for printing
        losses = estimator.get_histories('gate.loss')
        total_losses = None
        for loss_history in losses.values():
            if total_losses is None:
                total_losses = list(loss_history)
            else:
                total_losses = [
                    a + b for a, b in zip(total_losses, loss_history)]
        if total_losses is None:
            loss_mean = 0
        else:
            loss_mean = np.mean(total_losses)
        if loss_mean > 0:
            loss_std = Percent(np.std(total_losses) / loss_mean)
        else:
            loss_std = '?%'
        return 'gate.loss: {:.5f}±{}'.format(loss_mean, loss_std)

    @staticmethod
    def _gate_density_formatter(estimator):
        gates = estimator.get_values('gate.active')
        valid = total = 0
        for layer, gate in gates.items():
            valid += np.sum(gate.astype(np.float32) != 0)
            total += gate.size
        return 'gate: {}'.format(Percent(valid / total))

    @memoize_method
    def _register_gate_formatters(self):
        self.session.estimator.register_formatter(self._gate_loss_formatter)
        self.session.estimator.register_formatter(self._gate_density_formatter)

    def instantiate_gated_convolution(self, node, tensor, params):
        # register gate sparsity for printing
        self._register_gate_formatters()
        # params
        gate_params = {
            'density': params.pop('density'),
            'granularity': params.pop('granularity', 'channel'),
            'pool': params.pop('pool', 'max'),
            'policy': params.pop('policy', 'parametric_gamma'),
            'weight': params.pop('weight', 0.01),
            'squeeze_factor': params.pop('squeeze_factor', None),
            'should_gate': params.pop('should_gate', True),
        }
        return GatedConvolutionInstantiator(
            self, node, params, gate_params, tensor).instantiate()
