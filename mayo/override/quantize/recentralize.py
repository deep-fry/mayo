import tensorflow as tf

from mayo.util import memoize_property, object_from_params
from mayo.override import util
from mayo.override.base import OverriderBase, Parameter


class Recentralizer(OverriderBase):
    """ Recentralizes the distribution of pruned weights.  """

    class QuantizedParameter(Parameter):
        def __get__(self, instance, owner):
            if instance is None:
                return self
            var = super().__get__(instance, owner)
            return instance._quantize(var, mean_quantizer=True)

    positives = Parameter('positives', None, None, 'bool')
    positives_mean = QuantizedParameter('positives_mean', 1, [], 'float')
    negatives_mean = QuantizedParameter('negatives_mean', -1, [], 'float')

    def __init__(
            self, session, quantizer, mean_quantizer=None, should_update=True):
        super().__init__(session, should_update)
        cls, params = object_from_params(quantizer)
        self.quantizer = cls(session, **params)
        self.mean_quantizer = None
        if mean_quantizer:
            cls, params = object_from_params(mean_quantizer)
            self.mean_quantizer = cls(session, **params)

    @memoize_property
    def negatives(self):
        return util.logical_not(self.positives)

    def assign_parameters(self):
        super().assign_parameters()
        self.quantizer.assign_parameters()
        if self.mean_quantizer:
            self.mean_quantizer.assign_parameters()

    def _quantize(self, value, mean_quantizer=False):
        quantizer = self.mean_quantizer if mean_quantizer else self.quantizer
        quantizer = quantizer or self.quantizer
        scope = '{}/{}'.format(self._scope, self.__class__.__name__)
        if mean_quantizer and self.mean_quantizer:
            scope = '{}/mean'.format(scope)
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
        quantized = self._quantize(
            non_zeros * (positives_centralized + negatives_centralized))
        value = non_zeros * positives * (quantized + self.positives_mean)
        value += non_zeros * negatives * (quantized + self.negatives_mean)
        return value

    def _update(self):
        # update positives mask and mean values
        value = self.eval(self.before)
        # divide them into two groups
        # mean = util.mean(value)
        mean = 0.0
        # find two central points
        positives = value > mean
        self.positives = positives
        self.positives_mean = util.mean(value[util.where(positives)])
        negatives = util.logical_and(util.logical_not(positives), value != 0)
        self.negatives_mean = util.mean(value[util.where(negatives)])
        # update internal quantizer
        self.quantizer.update()

    def _info(self):
        info = self.quantizer.info()._asdict()
        if self.mean_quantizer:
            mean_info = self.mean_quantizer.info()
            for key, value in mean_info._asdict().items():
                info['mean_' + key] = value
        return self._info_tuple(**info)
