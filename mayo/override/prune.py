import math

import tensorflow as tf
import numpy as np

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
            self, session, alpha=None, on_factor=1.1, off_factor=0.9,
            should_update=True):
        super().__init__(session, alpha, should_update)
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

    @classmethod
    def finalize_info(cls, table):
        densities = table.get_column('density')
        count = table.get_column('count_')
        avg_density = sum(d * c for d, c in zip(densities, count)) / sum(count)
        footer = [None, '    overall: ', Percent(avg_density), None]
        table.set_footer(footer)
        return footer


class NetworkSlimmer(ChannelPrunerBase):
    """
    Our implementation of network slimming.

    density: the density of the slimmed feature maps.
    weight: the weight of L1 regularizer applied on BN gamma values.
    global_threshold:
        if true, uses all gamma values across layers overridden
        with NetworkSlimmer with should_update enabled.
    incremental:
        if true, .update() performs pruning on active channels, always
        decreases the overall density by a factor of (1 - density); otherwise,
        prune to the set density for all channels with a chance of re-enabling
        pruned channels.
    """
    def __init__(
            self, session, density, weight=0.01,
            global_threshold=True, incremental=False, should_update=True):
        super().__init__(session, should_update)
        self.density = density
        self.weight = weight
        self.global_threshold = global_threshold
        self.incremental = incremental

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
        tf.losses.add_loss(
            self.weight * tf.reduce_sum(tf.abs(gamma)),
            loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
        return util.cast(masked, float)

    def _threshold(self, values):
        if not values:
            return 0
        num_active = math.ceil(len(values) * self.density)
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
        gammas = []
        for overrider, gamma in estimator.get_values(gamma_name).items():
            if not overrider.should_update:
                continue
            if self.incremental:
                mask = self.session.run(overrider.mask)
                gamma = gamma[util.nonzero(mask)]
            gammas += gamma.tolist()
        threshold = self._threshold(gammas)
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
            if self.incremental:
                gammas = gamma[util.nonzero(self.mask)]
            threshold = self._threshold(gammas)
        new_mask = gamma > threshold
        if self.incremental:
            return util.logical_and(mask, new_mask)
        return new_mask


class FilterPruner(PrunerBase):
    density = Parameter('density', 0.0, [], tf.float32)
    mask = Parameter('mask', None, None, tf.bool)

    def __init__(self, session, density=None, should_update=True):
        super().__init__(session, should_update)
        self.density = density

    def _apply(self, value):
        self._parameter_config = {
            'mask': {
                'initial': tf.ones_initializer(dtype=tf.bool),
                'shape': tf.TensorShape([value.shape[-2], value.shape[-1]]),
            }
        }
        return value * util.cast(self.mask, float)

    def _l1_norm(self, value):
        # compute l1 norm for each filter
        axes = len(value.shape)
        assert axes == 4
        # mean, var = tf.nn.moments(util.abs(tensor), axes=[0, 1])
        # mean = np.mean(value, axis=(0, 1))
        # var = np.var(value, axis=(0, 1))
        # return mean + util.sqrt(var)
        return np.sum(util.abs(value), axis=(0, 1))

    def _threshold(self, value, density):
        value = value.flatten()
        index = int(value.size * density)
        return sorted(value)[index]

    def _updated_mask(self, tensor, mask):
        value, mask, density = self.session.run([tensor, mask, self.density])
        l1_norm = self._l1_norm(value)
        # mean, var = tf.nn.moments(util.abs(tensor), axes=[0, 1])
        return l1_norm > self._threshold(l1_norm, density)

    def _info(self):
        _, mask, density, count = super()._info()
        density = self.session.run(self.density)
        return self._info_tuple(
            mask=mask, density=density, count_=count)

    @classmethod
    def finalize_info(cls, table):
        footer = super().finalize_info(table)
        table.set_footer([None] + footer)
