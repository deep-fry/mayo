import math

import tensorflow as tf

from mayo.log import log
from mayo.util import memoize_method


class GateError(Exception):
    """Gating-related exceptions.  """


class GateParameterValueError(GateError):
    """Incorrect parameters used.  """


class GateGranularityTypeError(GateError):
    """Incorrect granularity used.  """


class GatedConvolutionBase(object):
    def __init__(
            self, constructor, node, conv_params, gate_params, conv_input):
        super().__init__()
        self.constructor = constructor
        self.is_training = constructor.is_training
        self.node = node
        self._init_convolution(conv_input, conv_params)
        self._init_gate_params(gate_params)

    def _init_convolution(self, tensor, params):
        self.kernel_size = params['kernel_size']
        self.stride = params.get('stride', 1)
        self.padding = params.get('padding', 'SAME')
        self.num_outputs = params['num_outputs']
        self.scope = params['scope']
        # delay normalization
        self.normalizer_fn = params.pop('normalizer_fn', None)
        self.normalizer_params = params.pop('normalizer_params', None)
        if self.normalizer_fn:
            # disable bias
            params['biases_initializer'] = None
        # delay activation
        self.activation_fn = params.get('activation_fn', tf.nn.relu)
        params['activation_fn'] = None
        # instantiate convolution
        self.input = tensor
        self.conved = self.constructor.instantiate_convolution(
            self.node, tensor, params)

    _must = object()
    _defaults = {
        'enable': True,
        'density': _must,
        'pool': 'avg',
        'granularity': 'channel',
        'weight': 0,
        'factor': 0,
        'trainable': True,
    }

    def _update_defaults(self, defaults):
        pass

    def _init_gate_params(self, params):
        self._update_defaults(self._defaults)
        # gate params
        for key, default in self._defaults.items():
            value = params.get(key, default)
            if value is self._must:
                raise KeyError(
                    'Gate parameter {!r} must be specified.'.format(key))
            setattr(self, key, value)
        self._check_granularity()

    def _check_granularity(self):
        accepted_granularities = ['channel', 'vector']
        if self.granularity in accepted_granularities:
            return
        raise GateGranularityTypeError(
            'Unrecognized granularity {}, we accept one of {}.'
            .format(self.granularity, ', '.join(self._accepted_granularities)))

    def subsample(self, tensor):
        """
        Subsample an input tensor, often the input tensor is
        the input to convolution.
        """
        num, height, width, channels = tensor.shape
        if self.granularity == 'channel':
            kernel = [height, width]
        elif self.granularity == 'vector':
            kernel = [1, width]
        else:
            self._check_granularity()
        pool_params = {
            'padding': 'VALID',
            'kernel_size': kernel,
            'stride': 1,
            'scope': '{}/subsample'.format(self.scope),
        }
        # subsampling
        if self.pool == 'max':
            # max pool is hardware-friendlier
            subsampled = self.constructor.instantiate_max_pool(
                None, tensor, pool_params)
        elif self.pool in ('l1', 'l2', 'avg'):
            if self.pool == 'l1':
                tensor = tf.abs(tensor)
            elif self.pool == 'l2':
                tensor = tf.square(tensor)
            subsampled = self.constructor.instantiate_average_pool(
                None, tensor, pool_params)
        else:
            raise GateParameterValueError(
                'feature extract type not supported.')
        # validate subsampled image
        num, height, width, channels = subsampled.shape
        if self.granularity == 'channel' and not (height == width == 1):
            raise GateParameterValueError(
                'We expect subsampled image for channel granularity '
                'to be 1x1.')
        if self.granularity == 'vector' and width != 1:
            raise GateParameterValueError(
                'We expect subsampled width for vector granularity to be 1.')
        return subsampled

    def _predictor(self, name):
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
        scope = '{}/{}'.format(self.scope, name)
        params = {
            'kernel_size': kernel,
            'stride': stride,
            'padding': padding,
            'num_outputs': self.num_outputs,
            'biases_initializer':
                tf.constant_initializer(0.0 if name == 'gate/beta' else 1.0),
            'weights_initializer':
                tf.truncated_normal_initializer(stddev=0.01),
            'normalizer_fn': None,  # potential FIXME no normalization
            'activation_fn': self.activation_fn,
            'scope': scope,
            'trainable': self.trainable,
        }
        subsampled = self.constructor.instantiate_numeric_padding(
            None, subsampled, params)
        # 2-layers of FC
        if self.factor > 0:
            intermediate_params = dict(params, **{
                'num_outputs': math.ceil(self.num_outputs / self.factor),
                'scope': '{}/factor'.format(scope),
            })
            subsampled = self.constructor.instantiate_convolution(
                None, subsampled, intermediate_params)
        return self.constructor.instantiate_convolution(
            None, subsampled, params)

    def _register(self, name, tensor):
        history = None if self.is_training else 'infinite'
        self.constructor.session.estimator.register(
            tensor, 'gate.{}'.format(name), self.node, history=history)
        return tensor

    @memoize_method
    def gate(self):
        tensor = self._predictor('gate')
        self._register('gamma', tensor)
        return tensor

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
        # reshape
        num, height, width, channels = tensor.shape
        flattened = tf.reshape(tensor, [num, -1])
        # number of active elements
        num_elements = int(flattened.shape[-1])
        num_active = math.ceil(num_elements * self.density)
        if num_active == num_elements:
            # all active, not gating
            return tf.ones(tensor.shape)
        # top_k, where k is the number of active channels
        top, _ = tf.nn.top_k(flattened, k=(num_active + 1))
        # disable channels with smaller activations
        threshold = tf.reduce_min(top, axis=[1], keep_dims=True)
        active = tf.reshape(flattened > threshold, tensor.shape)
        active = tf.stop_gradient(active)
        # register to estimator
        self._register('active', active)
        return active

    def normalize(self, tensor):
        if not self.normalizer_fn:
            return tensor
        # default normalization
        with tf.variable_scope(self.scope):
            return self.normalizer_fn(tensor, **self.normalizer_params)

    def activate(self, tensor):
        if not self.activation_fn:
            return tensor
        return self.activation_fn(tensor)

    def _add_regularization(self, loss):
        loss *= self.weight
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, loss)
        self.constructor.session.estimator.register(
            loss, 'gate.loss', self.node)

    def regularize(self):
        raise NotImplementedError

    def instantiate(self):
        self.normalized = self.normalize(self.conved)
        self.activated = self.activate(self.normalized)
        # regularize
        if self.weight <= 0:
            log.debug(
                'No regularizer used for {!r} in {!r}.'
                .format(self.__class__, self.node.formatted_name()))
        self.regularize()
        return self.activated


class SparseRegularizedGatedConvolutionBase(GatedConvolutionBase):
    def _update_defaults(self, defaults):
        defaults['regularizer'] = 'l1'
        defaults['epsilon'] = 0.001

    def _mixture(self, tensor, axes):
        mean, variance = tf.nn.moments(tensor, axes=axes)
        return variance / tf.square((mean + self.epsilon))

    def regularize(self):
        """
        We use a L1, L2 or MoE regularizer to encourage sparsity in gate.
        """
        regularizer = self.regularizer
        if isinstance(regularizer, str):
            regularizer = [regularizer]
        sparse = self.gate() * self.actives()
        loss = []
        if 'l1' in self.regularizer:
            loss.append(tf.abs(sparse))
        if 'l2' in self.regularizer:
            loss.append(tf.square(sparse))
        if 'moe' in self.regularizer:
            # mixture of experts
            loss.append(self._mixture(sparse, [0, 1, 2]))
        if 'moi' in self.regularizer:
            # mixture of idiots
            loss.append(self._mixture(sparse, [1, 2, 3]))
        loss = tf.add_n([tf.reduce_sum(l) for l in loss])
        self._add_regularization(loss)
