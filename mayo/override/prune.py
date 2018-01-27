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


class ChannelPruner(PrunerBase):
    # This pruner only works on activations
    # density = Parameter('density', 1.0, [], tf.float32)
    scaling_factors = Parameter(
        'scaling_factors', None, None, tf.float32, trainable=True)

    def __init__(self, density=None, should_update=True, weight=0.01):
        super().__init__(should_update)
        self.density = density
        self.weight = weight

    def _apply(self, value):
        masked = super()._apply(value)
        channel_shape = value.shape[3]
        self._parameter_config = {
            'scaling_factors': {
                'initial': tf.ones_initializer(dtype=tf.float32),
                'shape': channel_shape
            }
        }
        # add reg
        if self._has_batch_norm(value):
            gamma = self._pick_gamma(value)
            if gamma is None:
                raise ValueError(
                    'No BatchNorm scaling factors found for {}'.format(
                        value.name))
            loss = self.weight * tf.reduce_sum(tf.abs(gamma))
        else:
            loss = self.weight * tf.reduce_sum(tf.abs(self.scaling_factors))
        tf.losses.add_loss(
            loss, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
        if self._has_batch_norm(value):
            return masked
        return masked * self.scaling_factors

    def _pick_gamma(self, value):
        layer = value.name.split('/')[2]
        for variable in tf.trainable_variables():
            if layer in variable.name and 'BatchNorm/gamma' in variable.name:
                return variable
        return None

    def _has_batch_norm(self, value):
        return 'BatchNorm' in value.name

    def _updated_mask(self, var, mask, session):
        if self._has_batch_norm(var):
            scaling_factors = self._pick_gamma(var)
        else:
            scaling_factors = self.scaling_factors
        mask, scaling_factors = session.run(
            [mask, scaling_factors])
        chosen = int(len(scaling_factors) * self.density)
        sorted_factors = sorted(scaling_factors)
        threshold = sorted_factors[len(sorted_factors) - chosen - 1]
        # top_k, where k is the number of active channels
        # disable channels with smaller activation,
        (n, h, w, c) = mask.shape
        reshaped = mask.reshape((-1, c))
        reshaped = reshaped * util.cast((scaling_factors > threshold), float)
        return reshaped.reshape((n, h, w, c))

    def _info(self, session):
        _, mask, density, count = super()._info(session)
        return self._info_tuple(
            mask=mask, density=density, count_=count)

    @classmethod
    def finalize_info(cls, table):
        footer = super().finalize_info(table)
        table.set_footer([None] + footer)


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
