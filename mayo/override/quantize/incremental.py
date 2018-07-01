import tensorflow as tf
import numpy as np

from mayo.util import object_from_params
from mayo.override import util
from mayo.override.base import OverriderBase, Parameter


class IncrementalQuantizer(OverriderBase):
    """
    https://arxiv.org/pdf/1702.03044.pdf
    """
    interval = Parameter('interval', 0.1, [], 'float')
    mask = Parameter('mask', None, None, 'bool')

    def __init__(self, session, quantizer, interval, count_zero=True,
                 should_update=True):
        super().__init__(session, should_update)
        cls, params = object_from_params(quantizer)
        self.quantizer = cls(session, **params)
        self.count_zero = count_zero
        self.interval = interval

    def _quantize(self, value, mean_quantizer=False):
        quantizer = self.quantizer
        scope = '{}/{}'.format(self._scope, self.__class__.__name__)
        return quantizer.apply(self.node, scope, self._original_getter, value)

    def _apply(self, value):
        self._parameter_config = {
            'mask': {
                'initial': tf.zeros_initializer(tf.bool),
                'shape': value.shape,
            }
        }
        quantized_value = self._quantize(value)
        off_mask = util.cast(util.logical_not(self.mask), float)
        mask = util.cast(self.mask, float)
        # on mask indicates the quantized values
        return value * off_mask + quantized_value * mask

    def _policy(self, value, quantized, previous_mask, interval):
        previous_pruned = util.sum(previous_mask)
        if self.count_zero:
            th_arg = util.cast(util.count(value) * interval, int)
        else:
            tmp = util.count(value[value != 0])
            flat_value_arg = util.where(value.flatten() != 0)
            th_arg = util.cast(tmp * interval, int)
        if th_arg < 0:
            raise ValueError(
                'mask has {} elements, interval is {}'.format(
                    previous_pruned, interval))
        off_mask = util.cast(
            util.logical_not(util.cast(previous_mask, bool)), float)
        metric = value - quantized
        flat_value = (metric * off_mask).flatten()
        if interval >= 1.0:
            th = flat_value.max() + 1.0
        else:
            if self.count_zero:
                th = util.top_k(util.abs(flat_value), th_arg)
            else:
                th = util.top_k(util.abs(flat_value[flat_value_arg]), th_arg)
        th = util.cast(th, float)
        new_mask = util.logical_not(util.greater_equal(util.abs(metric), th))
        return util.logical_or(new_mask, previous_mask)

    # override assign_parameters to assign quantizer as well
    def assign_parameters(self):
        super().assign_parameters()
        self.quantizer.assign_parameters()

    def _update(self):
        # reset index
        self.quantizer.update()
        # if chosen quantized, change it to zeros
        value, quantized, mask, interval = self.session.run(
            [self.before, self.quantizer.after, self.mask, self.interval])
        new_mask = self._policy(value, quantized, mask, interval)
        self.session.assign(self.mask, new_mask)
