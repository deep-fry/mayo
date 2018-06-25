import math

import tensorflow as tf

from mayo import error
from mayo.util import memoize_method, memoize_property, null_scope
from mayo.net.graph import LayerNode


class GateError(error.MayoError):
    """Gating-related exceptions.  """


class GateParameterValueError(GateError):
    """Incorrect parameters used.  """


class GateGranularityTypeError(GateError):
    """Incorrect granularity used.  """


class GatedConvolutionBase(object):
    _must = object()
    _defaults = {
        'enable': True,
        'density': _must,
        'pool': 'avg',
        'granularity': 'channel',
        'regularizer': {},
        'factor': 0,
        'threshold': 'online',
        'decay': 0.9997,
        'trainable': True,
    }

    def __init__(
            self, constructor, node, conv_params, gate_params, conv_input):
        super().__init__()
        self.constructor = constructor
        self.estimator = constructor.session.estimator
        self.is_training = constructor.is_training
        self.node = node
        self._regularization_losses = []
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
        self.conv_params = params

    def _update_defaults(self, defaults):
        pass

    def _init_gate_params(self, params):
        self._update_defaults(self._defaults)
        # gate params
        for key, default in self._defaults.items():
            value = params.get(key, default)
            if value is self._must:
                raise error.KeyError(
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
                tf.truncated_normal_initializer(stddev=0.05),
                #  tf.contrib.layers.xavier_initializer(),
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
        self.estimator.register(
            tensor, 'gate.{}'.format(name), self.node, history=history)
        return tensor

    @memoize_method
    def gate(self):
        tensor = self._predictor('gate')
        self._register('gamma', tensor)
        return tensor

    @staticmethod
    def _threshold_debugger(estimator):
        info = {}
        for k, v in estimator.get_values('gate.threshold').items():
            if isinstance(k, LayerNode):
                k = k.formatted_name()
            info[k] = v
        return info

    @memoize_property
    def _threshold_variable(self):
        node = self.node if self.threshold == 'local' else None
        try:
            return self.estimator.get_tensor('gate.threshold', node)
        except KeyError:
            pass
        if self.threshold == 'local':
            scope = tf.variable_scope(self.scope)
        else:
            scope = null_scope()
        with scope:
            var = tf.get_variable(
                'gate/threshold', [],
                initializer=tf.constant_initializer(-0.1),
                trainable=False, dtype=tf.float32)
        try:
            self.estimator.get_tensor('gate.threshold')
        except KeyError:
            debug = {
                'formatter':
                    lambda e: 'gate.threshold: {:.4f}'.format(
                        e.get_value('gate.threshold')),
            }
        else:
            debug = {'debugger': self._threshold_debugger}
        self.estimator.register(var, 'gate.threshold', node=node, **debug)
        return var

    def _find_threshold(self, tensor):
        if tensor.shape.ndims != 2:
            tensor = tf.reshape(tensor, (tensor.shape[0], -1))
        num_elements = int(tensor.shape[-1])
        num_active = math.ceil(num_elements * self.density)
        if num_active == num_elements:
            # all active, not gating
            return None
        # top_k, where k is the number of active channels
        top, _ = tf.nn.top_k(tensor, k=(num_active + 1))
        # disable channels with smaller responses
        return tf.reduce_min(top, axis=[1], keepdims=True)

    def _finalizer(self):
        if self.threshold == 'online':
            return
        elif self.threshold == 'global':
            outputs = []
            gammas = self.estimator.get_tensors('gate.gamma')
            for node, tensor in gammas.items():
                if node.params['gate_params'].get('enable', True):
                    outputs.append(tensor)
            if not outputs:
                raise error.ValueError(
                    'Gated convolution did not register any gammas '
                    'for thresholding.')
            outputs = tf.concat(outputs, axis=-1)
        elif self.threshold == 'local':
            outputs = self.gate()
        else:
            raise GateParameterValueError('Unexpected threshold type.')
        threshold = self._find_threshold(outputs)
        if threshold is None:
            return
        threshold = tf.reduce_mean(threshold)
        # exponential moving average
        update_op = tf.assign(
            self._threshold_variable,
            self.decay * self._threshold_variable +
            (1 - self.decay) * threshold)
        ops = self.constructor.session.extra_train_ops
        if self.threshold == 'global':
            ops['gate'] = update_op
        else:
            ops = ops.setdefault('gate', {})
            ops[self.node] = update_op

    def bool_actives(self):
        if not (0 < self.density <= 1):
            raise GateParameterValueError(
                'Gate density value {} is out of range (0, 1].'
                .format(self.density))
        # not training with the output as we train the predictor `gate`
        tensor = tf.stop_gradient(self.gate())
        # reshape
        num, height, width, channels = tensor.shape
        flattened = tf.reshape(tensor, [num, -1])
        # find threshold
        if self.threshold == 'online':
            threshold = self._find_threshold(flattened)
            if threshold is None:
                return tf.ones(tensor.shape, dtype=tf.float32)
        elif self.threshold in ['local', 'global']:
            threshold = self._threshold_variable
            node = self.node if self.threshold == 'local' else 'gate'
            self.constructor.session.finalizers[node] = self._finalizer
        else:
            raise GateParameterValueError('Unexpected threshold type.')
        active = tf.reshape(flattened > threshold, tensor.shape)
        active = tf.stop_gradient(active)
        # register to estimator
        self._register('active', active)
        return active

    @memoize_method
    def actives(self):
        """
        Mark a portion of top elements in gate output to true,
        where the portion is approximately the specified density.
        """
        return tf.cast(self.bool_actives(), dtype=tf.float32)

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

    def _add_regularization(self, loss, weight):
        if weight <= 0:
            return
        loss *= weight
        self._regularization_losses.append(loss)

    def regularize(self):
        raise error.NotImplementedError

    def _instantiate_regularization(self):
        losses = []
        for loss in self._regularization_losses:
            if loss.shape.num_elements() > 1:
                loss = tf.reduce_sum(loss)
            losses.append(loss)
        if not losses:
            return
        loss = tf.add_n(losses)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, loss)
        self.estimator.register(loss, 'gate.loss', self.node)

    def instantiate(self):
        self.normalized = self.normalize(self.conved)
        self.activated = self.activate(self.normalized)
        # regularize
        self.regularize()
        self._instantiate_regularization()
        return self.activated
