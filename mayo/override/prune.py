import math

import tensorflow as tf

from mayo.log import log
from mayo.util import Percent, memoize_property
from mayo.override import util
from mayo.override.base import OverriderBase, Parameter


class PrunerBase(OverriderBase):
    mask = Parameter('mask', None, None, tf.bool)

    def __init__(self, session, should_update=True):
        super().__init__(session, should_update)

    def _apply(self, value):
        self._parameter_config = {
            'mask': {
                'initial': tf.ones_initializer(dtype=tf.bool),
                'shape': value.shape,
            }
        }
        return value * util.cast(self.mask, float)

    def _updated_mask(self, var, mask):
        raise NotImplementedError(
            'Method to compute an updated mask is not implemented.')

    def _update(self):
        mask = self._updated_mask(self.before, self.mask)
        self.session.assign(self.mask, mask)

    def _info(self):
        mask = util.cast(self.session.run(self.mask), int)
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

    def __init__(self, session, alpha=None, should_update=True):
        super().__init__(session, should_update)
        self.alpha = alpha

    def _threshold(self, tensor, alpha):
        # axes = list(range(len(tensor.get_shape()) - 1))
        tensor_shape = util.get_shape(tensor)
        axes = list(range(len(tensor_shape)))
        mean, var = util.moments(util.abs(tensor), axes)
        if alpha is None:
            return mean + self.alpha * util.sqrt(var)
        return mean + alpha * util.sqrt(var)

    def _updated_mask(self, var, mask):
        return util.abs(var) > self._threshold(var)

    def _info(self):
        _, mask, density, count = super()._info()
        alpha = self.session.run(self.alpha)
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

    def _updated_mask(self, var, mask):
        var, mask, alpha = self.session.run([var, mask, self.alpha])
        threshold = self._threshold(var, alpha)
        on_mask = util.abs(var) > self.on_factor * threshold
        mask = util.logical_or(mask, on_mask)
        off_mask = util.abs(var) > self.off_factor * threshold
        # import pdb; pdb.set_trace()
        return util.logical_and(mask, off_mask)


class ChannelPrunerBase(OverriderBase):
    mask = Parameter('mask', None, None, tf.bool)

    def __init__(self, session, should_update=True):
        super().__init__(session, should_update)

    def _apply(self, value):
        # check shape
        if not len(value.shape) >= 3:
            raise ValueError(
                'Incorrect dimension {} for channel pruner'
                .format(value.shape))
        self.num_channels = value.shape[-1]
        self._parameter_config = {
            'mask': {
                'initial': tf.ones_initializer(dtype=tf.bool),
                'shape': self.num_channels,
            },
        }
        mask = self.mask
        for _ in range(3):
            mask = tf.expand_dims(mask, 0)
        return value * util.cast(mask, float)

    def _updated_mask(self, var, mask):
        raise NotImplementedError(
            'Method to compute an updated mask is not implemented.')

    def _update(self):
        mask = self._updated_mask(self.before, self.mask)
        self.session.assign(self.mask, mask)

    def _info(self):
        mask = util.cast(self.session.run(self.mask), int)
        density = Percent(util.sum(mask) / util.count(mask))
        return self._info_tuple(
            mask=self.mask.name, density=density, count_=mask.size)


class NetworkSlimmer(ChannelPrunerBase):
    def __init__(
            self, session, density=None, weight=0.01,
            global_threshold=True, should_update=True):
        super().__init__(session, should_update)
        self.density = density
        self.weight = weight
        self.global_threshold = global_threshold

    @memoize_property
    def gamma(self):
        name = '{}/BatchNorm/gamma'.format(self.node.formatted_name())
        trainables = tf.trainable_variables()
        for v in trainables:
            if v.op.name == name:
                return v
        raise ValueError(
            'Unable to find gamma {!r} for layer {!r}.'
            .format(name, self.node.formatted_name()))

    def _apply(self, value):
        masked = super()._apply(value)
        gamma = self.gamma
        # register the latest gamma and mask to be used for later update
        # TODO this works as a way to collect global gammas, but the `gamma`
        # tensor is evaluated every time we use `session.run(batch=True)`,
        # will fix later if performance proves to be problematic.
        self.session.estimator.register(
            gamma, 'NetworkSlimmer.gamma', node=self, history=1)
        # add reg
        loss = self.weight * tf.nn.l2_loss(gamma)
        tf.losses.add_loss(
            # self.weight * tf.reduce_sum(tf.abs(gamma)),
            loss,
            loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
        return util.cast(masked, float)

    def _threshold(self, values, non_zeros, total):
        if not values:
            return 0
        # density = (previous_zeros + new_zeros) * total
        density = self.density - non_zeros / float(total)
        num_active = math.ceil(len(values) * density)
        if num_active == len(values):
            return 0
        return sorted(values)[-num_active - 1]

    def _global_threshold(self):
        estimator = self.session.estimator
        gamma_name = 'NetworkSlimmer.gamma'
        threshold_name = 'NetworkSlimmer.threshold'
        if estimator.max_len(gamma_name) == 0:
            try:
                return estimator.get_value(threshold_name)
            except KeyError:
                raise RuntimeError(
                    'Train for a while before running update to collect '
                    'gamma values.')
        # extract all gammas globally
        gammas = estimator.get_values(gamma_name)
        gamma_values = []
        non_zeros = total = 0
        for overrider, gamma in gammas.items():
            mask = self.session.run(overrider.mask)
            gamma_values += gamma[util.nonzero(mask)].tolist()
            non_zeros += util.sum(mask)
            total += len(gamma)
        threshold = self._threshold(gamma_values, non_zeros, total)
        log.debug(
            'Extracted a global threshold for all gammas: {}.'
            .format(threshold))
        estimator.flush_all(gamma_name)
        estimator.add(threshold, threshold_name)
        return threshold

    def _updated_mask(self, var, mask):
        mask, gamma = self.session.run([mask, self.gamma])
        if self.global_threshold:
            threshold = self._global_threshold()
        else:
            gamma_values = gamma[util.nonzero(self.mask)]
            threshold = self._threshold(gamma_values)
        return util.logical_and(mask, gamma > threshold)


class FilterPruner(PrunerBase):
    # TODO: finish channel pruner
    alpha = Parameter('alpha', -2, [], tf.float32)

    def __init__(self, session, alpha=None, should_update=True):
        super().__init__(session, should_update)
        self.alpha = alpha

    def _threshold(self, tensor):
        axes = len(tensor.get_shape())
        assert axes == 4
        mean, var = tf.nn.moments(util.abs(tensor), axes=[1, 2])
        return mean + self.alpha * util.sqrt(var)

    def _updated_mask(self, var, mask):
        return util.abs(var) > self._threshold(var)

    def _info(self):
        _, mask, density, count = super()._info()
        alpha = self.session.run(self.alpha)
        return self._info_tuple(
            mask=mask, alpha=alpha, density=density, count_=count)

    @classmethod
    def finalize_info(cls, table):
        footer = super().finalize_info(table)
        table.set_footer([None] + footer)
