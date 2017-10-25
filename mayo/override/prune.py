import tensorflow as tf

from mayo.log import log
from mayo.util import Percent
from mayo.override import util
from mayo.override.base import OverriderBase


class PrunerBase(OverriderBase):
    def _apply(self, getter, value):
        shape = value.get_shape()
        name = '{}/mask'.format(value.op.name)
        self._mask = getter(
            name, dtype=tf.bool, shape=shape,
            initializer=tf.ones_initializer(), trainable=False)
        mask = util.cast(self._mask, float)
        return value * mask

    def _updated_mask(self, var, mask, session):
        raise NotImplementedError(
            'Method to compute an updated mask is not implemented.')

    def _update(self, session):
        mask = self._updated_mask(self.before, self._mask, session)
        return session.run(tf.assign(self._mask, mask))

    def info(self, session):
        mask = util.cast(session.run(self._mask), int)
        density = Percent(util.sum(mask) / util.count(mask))
        return self._info_tuple(
            mask=self._mask.name, density=density, count_=mask.size)

    @classmethod
    def finalize_info(cls, table):
        densities = table.get_column('density')
        count = table.get_column('count_')
        avg_density = sum(d * c for d, c in zip(densities, count)) / sum(count)
        table.set_footer([None, '    overall: ', Percent(avg_density), None])


class ThresholdPruner(PrunerBase):
    def __init__(self, threshold, should_update=True):
        super().__init__(should_update)
        self.threshold = threshold

    def _updated_mask(self, var, mask, session):
        return util.binarize(var, self.threshold)


class MeanStdPruner(PrunerBase):
    def __init__(self, alpha, should_update=True):
        super().__init__(should_update)
        self.alpha = alpha

    def _threshold(self, tensor):
        # axes = list(range(len(tensor.get_shape()) - 1))
        axes = list(range(len(tensor.get_shape())))
        mean, var = tf.nn.moments(util.abs(tensor), axes=axes)
        return mean + self.alpha * util.sqrt(var)

    def _updated_mask(self, var, mask, session):
        return util.binarize(var, self._threshold(var))


class DynamicNetworkSurgeryPruner(MeanStdPruner):
    """
    References:
        1. https://github.com/yiwenguo/Dynamic-Network-Surgery
        2. https://arxiv.org/abs/1608.04493
    """
    def __init__(
            self, c_rate, on_factor=1.1, off_factor=0.9, should_update=True):
        super().__init__(c_rate, should_update)
        self.on_factor = on_factor
        self.off_factor = off_factor

    def _updated_mask(self, var, mask, session):
        threshold = self._threshold(var)
        on_mask = util.abs(var) > self.on_factor * threshold
        mask = util.logical_or(mask, on_mask)
        off_mask = util.abs(var) > self.off_factor * threshold
        return util.logical_and(mask, off_mask)


class MayoDNSPruner(DynamicNetworkSurgeryPruner):
    def __init__(
            self, c_rate, on_factor=1.1, off_factor=0.9, should_update=True):
        super().__init__(c_rate, on_factor, off_factor, should_update)

    def _updated_mask(self, var, mask, session):
        return super()._updated_mask(var, mask, session)

    def _threshold_update(self):
        self.alpha += self.scale
        log.info('inside overrider, alpha is {}'.format(self.alpha))

    def _scale_roll_back(self):
        self.alpha -= self.scale

    def _scale_update(self, update_factor):
        self.scale = self.scale * update_factor

    def _setup(self, session):
        self.scale = session.config.retrain.scale
