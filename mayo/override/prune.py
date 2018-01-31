import math

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
        session.assign(self.mask, mask)

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


class MeanStdPruner(PrunerBase):
    alpha = Parameter('alpha', -2, [], tf.float32)

    def __init__(self, alpha=None, should_update=True):
        super().__init__(should_update)
        self.alpha = alpha

    def _threshold(self, tensor, alpha):
        # axes = list(range(len(tensor.get_shape()) - 1))
        tensor_shape = util.get_shape(tensor)
        axes = list(range(len(tensor_shape)))
        mean, var = util.moments(util.abs(tensor), axes)
        if alpha is None:
            return mean + self.alpha * util.sqrt(var)
        return mean + alpha * util.sqrt(var)

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
        var, mask, alpha = session.run([var, mask, self.alpha])
        threshold = self._threshold(var, alpha)
        on_mask = util.abs(var) > self.on_factor * threshold
        mask = util.logical_or(mask, on_mask)
        off_mask = util.abs(var) > self.off_factor * threshold
        # import pdb; pdb.set_trace()
        return util.logical_and(mask, off_mask)


class ChannelPrunerBase(OverriderBase):
    channel_mask = Parameter('channel_mask', None, None, tf.bool)

    def __init__(self, should_update=True):
        super().__init__(should_update)

    def _apply(self, value):
        # check shape
        if not len(value.shape) >= 3:
            raise ValueError('Incorrect dimension {} for channel pruner'
                             .format(value.shape))
        self.channel_shape = value.shape[3]
        self._parameter_config = {
            'channel_mask': {
                'initial': tf.ones_initializer(dtype=tf.bool),
                'shape': self.channel_shape,
            }
        }
        mask = tf.expand_dims(self.channel_mask, 0)
        return value * util.cast(mask, float)

    def _updated_mask(self, var, mask, session):
        raise NotImplementedError(
            'Method to compute an updated mask is not implemented.')

    def _update(self, session):
        channel_mask = self._updated_mask(
            self.before, self.channel_mask, session)
        session.assign(self.channel_mask, channel_mask)


class ChannelPruner(ChannelPrunerBase):
    # This pruner only works on activations
    def __init__(self, density=None, weight=0.01, should_update=True):
        super().__init__(should_update)
        self.density = density
        self.weight = weight

    def _has_batch_norm(self, value):
        # FIXME a hack to determine this layer has batch norm
        return 'BatchNorm' in value.name

    def _compute_scale(self, value):
        layer = value.name.split('/')[2]
        for variable in tf.trainable_variables():
            if layer in variable.name and 'BatchNorm/gamma' in variable.name:
                return variable
        raise ValueError('Unable to find gamma from BatchNorm')

    def _apply(self, value):
        masked = super()._apply(value)
        self._parameter_config = {
            'scaling_factors': {
                'initial': tf.ones_initializer(dtype=tf.float32),
                'shape': self.channel_shape,
            }
        }
        # add reg
        self.scale = self._compute_scale(value)
        tf.losses.add_loss(
            self.weight * tf.reduce_sum(tf.abs(self.scale)),
            loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
        return util.cast(masked, float)

    def _updated_mask(self, var, channel_mask, session):
        channel_mask, scale = session.run([channel_mask, self.scale])
        num_active = math.ceil(len(scale) * self.density)
        threshold = sorted(scale)[-num_active]
        # top_k, where k is the number of active channels
        # disable channels with smaller activation,
        channel_mask = channel_mask * util.cast((scale > threshold), float)
        return channel_mask


class FilterPruner(PrunerBase):
    # TODO: finish channel pruner
    alpha = Parameter('alpha', -2, [], tf.float32)

    def __init__(self, alpha=None, should_update=True):
        super().__init__(should_update)
        self.alpha = alpha

    def _threshold(self, tensor):
        axes = len(tensor.get_shape())
        assert axes == 4
        mean, var = tf.nn.moments(util.abs(tensor), axes=[1, 2])
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
