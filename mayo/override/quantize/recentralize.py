import tensorflow as tf

from mayo.util import memoize_property, object_from_params
from mayo.override import util
from mayo.override.base import OverriderBase, Parameter
from mayo.log import log


class Recentralizer(OverriderBase):
    """ Recentralizes the distribution of pruned weights.  """

    class QuantizedParameter(Parameter):
        def _quantize(self, instance, value):
            scope = '{}/{}/{}'.format(
                instance._scope, instance.__class__.__name__, self.name)
            quantizer = instance.parameter_quantizers.get(self.name)
            if not quantizer:
                return value
            return quantizer.apply(
                instance.node, scope, instance._original_getter, value)

        def __get__(self, instance, owner):
            if instance is None:
                return self
            name = '_quantized_{}'.format(self.name)
            try:
                return instance._parameter_variables[name]
            except KeyError:
                pass
            var = super().__get__(instance, owner)
            var = self._quantize(instance, var)
            instance._parameter_variables[name] = var
            return var

    positives = Parameter('positives', None, None, 'bool')
    positives_mean = QuantizedParameter('positives_mean', 1, [], 'float')
    negatives_mean = QuantizedParameter('negatives_mean', -1, [], 'float')

    def __init__(
            self, session, quantizer, mean_quantizer=None,
            should_update=True, reg=False):
        super().__init__(session, should_update)
        cls, params = object_from_params(quantizer)
        self.quantizer = cls(session, **params)
        self.reg = reg
        if mean_quantizer:
            cls, params = object_from_params(mean_quantizer)
            self.parameter_quantizers = {
                'positives_mean': cls(session, **params),
                'negatives_mean': cls(session, **params),
            }
        else:
            self.parameter_quantizers = {}

    @memoize_property
    def negatives(self):
        return util.logical_not(self.positives)

    def assign_parameters(self):
        super().assign_parameters()
        self.quantizer.assign_parameters()
        for quantizer in self.parameter_quantizers.values():
            quantizer.assign_parameters()

    def _quantize(self, value):
        quantizer = self.quantizer
        scope = '{}/{}'.format(self._scope, self.__class__.__name__)
        return quantizer.apply(self.node, scope, self._original_getter, value)

    def _apply(self, value):
        # dynamic parameter configuration
        self._parameter_config = {
            'positives': {
                'initial': tf.ones_initializer(tf.bool),
                'shape': value.shape,
            },
        }
        positives = util.cast(self.positives, float)
        negatives = util.cast(self.negatives, float)
        non_zeros = util.cast(tf.not_equal(self.before, 0), float)

        positives_centralized = positives * (value - self.positives_mean)
        negatives_centralized = negatives * (value - self.negatives_mean)
        # keep a track of quantized value, without means
        self.quantized = self._quantize(
            non_zeros * (positives_centralized + negatives_centralized))
        quantized_value = non_zeros * positives * \
            (self.quantized + self.positives_mean)
        quantized_value += non_zeros * negatives * \
            (self.quantized + self.negatives_mean)

        self._quantization_loss_regularizer(value, quantized_value)
        return quantized_value

    def _quantization_loss_regularizer(self, value, quantized_value):
        if self.reg == 0.0:
            return
        loss = tf.reduce_sum(tf.abs(value - quantized_value))
        loss *= self.reg
        loss_name = tf.GraphKeys.REGULARIZATION_LOSSES
        tf.add_to_collection(loss_name, loss)

    def _update(self):
        # update positives mask and mean values
        value = self.session.run(self.before)
        # divide them into two groups
        # mean = util.mean(value)
        mean = 0.0
        # find two central points
        positives = value > mean
        self.positives = positives
        self.positives_mean = util.mean(value[util.where(positives)])
        negatives = util.logical_and(util.logical_not(positives), value != 0)
        self.negatives_mean = util.mean(value[util.where(negatives)])
        if self.positives_mean.eval() == 0 or self.negatives_mean.eval() == 0:
            log.warn('means are skewed, pos mean is {} and neg mean is {}'
                     .format(self.positives_mean.eval(),
                             self.negatives_mean.eval()))
        # update internal quantizer
        self.quantizer.update()
        for quantizer in self.parameter_quantizers.values():
            quantizer.update()

    def _info(self):
        info = self.quantizer.info()._asdict()
        for name, quantizer in self.parameter_quantizers.items():
            param_info = quantizer.info()
            param_info = {
                '{}_{}'.format(name, key): value
                for key, value in param_info._asdict().items()
            }
            info.update(param_info)
        return self._info_tuple(**info)
