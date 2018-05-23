import tensorflow as tf

from mayo.log import log
from mayo.util import memoize_property
from mayo.override import util
from mayo.override.prune.base import ChannelPrunerBase


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
        trainables = self.session.trainable_variables()
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
        num_active = util.ceil(len(values) * self.density)
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

    def estimate(self, layer_info, info):
        mask = [self.session.run(self.mask)]
        macs = layer_info.get('macs', 0)
        density = self.session.estimator._mask_density(mask)
        update = {
            '_mask': mask,
            'density': density,
            'macs': int(macs * density),
        }
        return layer_info.update(update)
