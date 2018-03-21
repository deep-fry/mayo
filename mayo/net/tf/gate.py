import math

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from mayo.log import log
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
    _accepted_policies = [
        'naive', 'parametric_gamma', 'squeeze_excitation']
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
        must = object()
        defaults = {
            'enable': True,
            'density': must,
            'policy': 'parametric_gamma',
            'pool': 'avg',
            'granularity': 'channel',
            'norm': 'batch',
            'weight': 0,
            'squeeze_factor': None,
            'trainable': True,
        }
        for key, default in defaults.items():
            value = gate_params.get(key, default)
            if value is must:
                raise KeyError(
                    'Gate parameter {!r} must be specified.'.format(key))
            setattr(self, key, value)
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
                raise GateParameterValueError(
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
        normalizer_fn = None
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
            'trainable': self.trainable,
        }
        padded = self.constructor.instantiate_numeric_padding(
            None, subsampled, params)
        return self.constructor.instantiate_convolution(None, padded, params)

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
            return tf.ones(tensor.shape)
            # return tensor
        # reshape the last dimensions into one
        reshaped = tf.reshape(tensor, [num, -1])
        # top_k, where k is the number of active channels
        top, _ = tf.nn.top_k(reshaped, k=(num_active + 1))
        # disable channels with smaller activations
        threshold = tf.reduce_min(top, axis=[1], keep_dims=True)
        active = tf.reshape(reshaped > threshold, tensor.shape)
        return tf.stop_gradient(active)

    def _gated_normalization(self, prenorm):
        # gating
        if self.enable:
            actives = tf.cast(self.actives(), tf.float32)
        else:
            actives = None
        if not self.normalizer_fn:
            if self.enable:
                return prenorm * actives
            return prenorm
        if self.policy == 'parametric_gamma':
            # output =
            #   actives(gamma(x)) * gamma(x) * norm(conv(x)) +
            #   actives(gamma(x)) * beta
            if not self.normalizer_params.get('scale', False):
                raise GateParameterValueError(
                    'Policy "parametric_gamma" expects `scale` to be used '
                    'in slim.batch_norm.')
            normalizer_params = dict(self.normalizer_params, **{
                'scale': False,
                'center': False,
                'activation_fn': None,
                'scope': '{}/BatchNorm'.format(self.scope),
                'is_training': self.is_training,
            })
            if self.norm == 'batch':
                output = self.constructor.instantiate_batch_normalization(
                    None, prenorm, normalizer_params)
            elif self.norm == 'channel':
                norm_mean, norm_var = tf.nn.moments(
                    prenorm, axes=[1, 2], keep_dims=True)
                output = (prenorm - norm_mean) / tf.sqrt(norm_var)
            else:
                raise GatePolicyTypeError('Unrecognized normalization policy.')
            gamma = self.gate()
            output *= actives * gamma if self.enable else gamma
            if not self.normalizer_params.get('center', True):
                return output
            # use offset beta
            beta_scope = '{}/gate/shift'.format(self.scope)
            beta = tf.get_variable(
                beta_scope, shape=output.shape[-1], dtype=tf.float32,
                initializer=tf.constant_initializer(0.1),
                trainable=self.trainable)
            # gate output is the parametric gamma value
            output += actives * beta if self.enable else beta
            return output
        # normal normalization
        scope = '{}/{}'.format(
            self.scope, self._normalizer_names[self.normalizer_fn])
        normalizer_params = dict(self.normalizer_params, scope=scope)
        output = self.normalizer_fn(self.output, **normalizer_params)
        return actives * output if self.enable else output

    def _squeeze_excitation(self, tensor):
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
        squeezed = conv(None, tensor, squeeze_params)
        expand_params = params('Expand', self.num_outputs)
        return conv(None, squeezed, expand_params)

    def instantiate(self):
        output = self._gated_normalization(self.output)
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        if self.policy == 'squeeze_excitation':
            se = self._squeeze_excitation(output)
            output *= se
        activated = output
        # estimator
        self._register()
        # regularizer
        if self.policy == 'naive':
            match = self.subsample(activated)
        elif self.policy == 'parametric_gamma':
            match = None
        elif self.policy == 'squeeze_excitation':
            match = se
        else:
            self._check_policy()
        self._regularize(self.gate(), match)
        return output

    def _regularize(self, gate, match):
        """
        Regularize gate by making gate output `gate` to match `match` as close
        as possible.  If `match` is not supplied, we use a L1 regularizer to
        drive `gate` to zero.
        """
        if self.weight <= 0:
            if match is not None:
                raise GateParameterValueError(
                    'We expect weight to be non-zero if `match` is specified, '
                    'as without `match` to regularize gate, the gate network '
                    'is not learning anything.')
        loss = None
        loss_name = tf.GraphKeys.REGULARIZATION_LOSSES
        if match is not None:
            # training
            # policy descriminator: we simply match values in each channel
            # using a loss regularizer
            loss = tf.losses.mean_squared_error(
                tf.stop_gradient(match), gate, loss_collection=None)
        else:
            # parametric gamma does not match anything
            epsilon = 0.01
            actives = self.actives()
            # one loss
            mean, variance = tf.nn.moments(gate * actives, axes=[0, 1, 2])
            loss = tf.reduce_sum(
                tf.square(tf.sqrt(variance) / (mean + epsilon)))
            # another loss
            mean, variance = tf.nn.moments(gate, axes=[1, 2, 3])
            loss += tf.reduce_sum(
                tf.square(tf.sqrt(variance) / (mean + epsilon)))
            # loss = tf.reduce_sum(tf.abs(gate))
        loss *= self.weight
        tf.add_to_collection(loss_name, loss)
        self.constructor.session.estimator.register(
            loss, 'gate.loss', self.node)

    def _register(self):
        history = None if self.is_training else 'infinite'
        self.constructor.session.estimator.register(
            self.gate(), 'gate.output', self.node, history=history)
        if self.enable:
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
        if math.isnan(loss_mean):
            log.error(
                'Gating loss is NaN. Please check your regularizer weight.')
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
        gate_params = params.pop('gate_params')
        return GatedConvolutionInstantiator(
            self, node, params, gate_params, tensor).instantiate()
