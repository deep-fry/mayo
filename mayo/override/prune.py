import tensorflow as tf

from mayo.util import Percent
from mayo.override import util
from mayo.override.base import OverriderBase, Parameter


class PrunerBase(OverriderBase):
    mask = Parameter('mask', None, None, tf.bool)

    def __init__(self, should_update=True):
        super().__init__(should_update)

    def _apply(self, value):
        self._parameter_config = {
            'mask': {
                'initial': tf.ones_initializer(dtype=tf.bool),
                'shape': value.shape,
            }
        }
        return value * util.cast(self.mask, float)

    def _updated_mask(self, var, mask, session):
        raise NotImplementedError(
            'Method to compute an updated mask is not implemented.')

    def _update(self, session):
        mask = self._updated_mask(self.before, self.mask, session)
        session.run(tf.assign(self.mask, mask))

    def _info(self, session):
        mask = util.cast(session.run(self.mask), int)
        density = Percent(util.sum(mask) / util.count(mask))
        return self._info_tuple(
            mask=self.mask.name, density=density, count_=mask.size)

    @classmethod
    def finalize_info(cls, table):
        densities = table.get_column('density')
        count = table.get_column('count_')
        avg_density = sum(d * c for d, c in zip(densities, count)) / sum(count)
        footer = [None, '    overall: ', Percent(avg_density), None]
        table.set_footer(footer)
        return footer


class RandomChannelPruner(OverriderBase):
    def __init__(self, ratio=None, should_update=True):
        super().__init__(should_update)
        self.ratio = ratio

    def _apply(self, session):
        return self._apply_value(self.before, session)

    def _update(self, session):
        return

    def _apply_value(self, var, session):
        n, h, w, c = var.shape
        n = int(n)
        c = int(c)
        # threshold
        omap = {'Sign': 'Identity'}
        random_number = tf.random_uniform(shape=[n, 1, 1, c],
            minval=self.ratio-1, maxval=self.ratio)
        with tf.get_default_graph().gradient_override_map(omap):
            self.gate = tf.sign(random_number)
            self.gate = tf.clip_by_value(self.gate, 0, 1)
        # gates out feature maps with low vairance and replace the whole feature
        # map with its mean
        tf.add_to_collection('mayo.gates', self.gate)
        return self.gate * var

    def _info(self, session):
        gate = util.cast(session.run(self.gate), int)
        density = Percent(util.sum(gate) / util.count(gate))
        return self._info_tuple(
            mask=self.gate.name, density=density, count_=gate.size)

    @classmethod
    def finalize_info(cls, table):
        densities = table.get_column('density')
        count = table.get_column('count_')
        avg_density = sum(d * c for d, c in zip(densities, count)) / sum(count)
        footer = [None, '    overall: ', Percent(avg_density), None]
        table.set_footer(footer)


class ChannelPruner(OverriderBase):
    threshold = Parameter('threshold', 1, [], tf.float32)

    def __init__(self, alpha=None, threshold=None, should_update=True):
        super().__init__(should_update)
        self.threshold = threshold

    def _apply(self, session):
        return self._apply_value(self.before, session)

    def _update(self, session):
        return

    def _apply_value(self, var, session):
        var_relu = tf.nn.relu(var)
        n, h, w, c = var.shape
        pool_params = {
            'padding': 'VALID',
            'ksize': [1, h, w, 1],
            'strides': [1, 1, 1, 1]
        }
        pooled = tf.nn.avg_pool(var_relu, **pool_params)

        n = int(n)
        c = int(c)
        mean, variance = tf.nn.moments(var, axes=[1, 2])
        variance = tf.reshape(variance, shape=[n, 1, 1, c])
        mean = tf.reshape(mean, shape=[n, 1, 1, c])
        # threshold
        # omap = {'Sign': 'Identity'}
        # with tf.get_default_graph().gradient_override_map(omap):
        #     self.gate = tf.sign(mean - self.threshold)
        #     self.gate = tf.clip_by_value(self.gate, 0, 1)
        # gates out feature maps with low vairance and replace the whole feature
        # map with its mean
        self.gate = util.cast(tf.abs(pooled) > self.threshold, float)
        self.pooled = pooled
        tf.add_to_collection('mayo.gates', self.gate)
        # return mean * (1 - self.gate) + self.gate * var
        return self.gate * var

    def _info(self, session):
        gate = util.cast(session.run(self.gate), int)
        density = Percent(util.sum(gate) / util.count(gate))
        return self._info_tuple(
            mask=self.gate.name, density=density, count_=gate.size)

    @classmethod
    def finalize_info(cls, table):
        densities = table.get_column('density')
        count = table.get_column('count_')
        avg_density = sum(d * c for d, c in zip(densities, count)) / sum(count)
        footer = [None, '    overall: ', Percent(avg_density), None]
        table.set_footer(footer)


class MeanStdPruner(PrunerBase):
    alpha = Parameter('alpha', 1, [], tf.float32)

    def __init__(self, alpha=None, should_update=True):
        super().__init__(should_update)
        self.alpha = alpha

    def _threshold(self, tensor):
        # axes = list(range(len(tensor.get_shape()) - 1))
        axes = list(range(len(tensor.get_shape())))
        mean, var = tf.nn.moments(util.abs(tensor), axes=axes)
        return mean + self.alpha * util.sqrt(var)

    def _updated_mask(self, var, mask, session):
        return util.abs(var) > self._threshold(var)

    def _info(self, session):
        _, mask, density, count = super()._info(session)
        alpha = session.run(self.alpha)
        return self._info_tuple(
            mask=mask, alpha=alpha, density=density, count_=count)

    @classmethod
    def finalize_info(cls, table):
        footer = super().finalize_info(table)
        table.set_footer([None] + footer)


class DynamicNetworkSurgeryPruner(MeanStdPruner):
    """
    References:
        1. https://github.com/yiwenguo/Dynamic-Network-Surgery
        2. https://arxiv.org/abs/1608.04493
    """
    def __init__(
            self, alpha=None, on_factor=1.1, off_factor=0.9,
            should_update=True):
        super().__init__(alpha, should_update)
        self.on_factor = on_factor
        self.off_factor = off_factor

    def _updated_mask(self, var, mask, session):
        threshold = self._threshold(var)
        on_mask = util.abs(var) > self.on_factor * threshold
        mask = util.logical_or(mask, on_mask)
        off_mask = util.abs(var) > self.off_factor * threshold
        return util.logical_and(mask, off_mask)
