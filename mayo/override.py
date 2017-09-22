import collections

import tensorflow as tf


def _round(value):
    omap = {'Round': 'Identity'}
    with tf.get_default_graph().gradient_override_map(omap):
        return tf.round(value)


def _binarize(tensor, threshold):
    return tf.cast(tf.abs(tensor) > threshold, tf.float32)


def _clip_by_value(tensor, minimum, maximum, transparent_backprop=False):
    omap = {}
    if transparent_backprop:
        omap = {'Minimum': 'Identity', 'Maximum': 'Identity'}
    with tf.get_default_graph().gradient_override_map(omap):
        return tf.clip_by_value(tensor, minimum, maximum)


class BaseOverrider(object):
    """
    Base class for applying overriding operations on a Net.  Please ensure
    both methods `_apply` and `_update` are overridden with appropriate
    implementations.

    The method `_apply` overrides the variable in `value`, returns the
    overridden result; `_update` updates states of tensorflow variables used in
    `_apply`.
    """
    class OverrideNotAppliedError(Exception):
        """Invoke apply before update.  """

    def _apply(self, getter, value):
        """
        Things to apply to the variable in `value`, returns the
        overridden result.
        """
        raise NotImplementedError(
            'Overrider method "apply" must be implemented.')

    def apply(self, getter, value):
        self._applied = True
        self._name = value.op.name
        self.before = value
        self.after = self._apply(getter, value)
        return self.after

    def _update(self):
        """
        Update things to apply during training, returns the update operation.
        """
        raise NotImplementedError(
            'Update operation not supported on this overrider class {!r}.'
            .format(self.__class__))

    def update(self):
        if not getattr(self, '_applied', False):
            raise self.__class__.OverrideNotAppliedError(
                '"apply" must be invoked before call "update".')
        return self._update()


class ChainOverrider(BaseOverrider, collections.Sequence):
    def __init__(self, overriders):
        super().__init__()
        self._overriders = overriders

    def __getitem__(self, index):
        return self._overriders[index]

    def __len__(self):
        return len(self._overriders)

    def _apply(self, getter, value):
        for o in self._overriders:
            value = o.apply(getter, value)
        return value

    def _update(self):
        ops = []
        for o in self._overriders:
            try:
                ops.append(o.update())
            except NotImplementedError:
                pass
        return tf.group(*ops)


class ThresholdBinarizer(BaseOverrider):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def _apply(self, _, value):
        return _binarize(value, self.threshold)


class BasePruner(BaseOverrider):
    def _apply(self, getter, value):
        shape = value.get_shape()
        name = '{}/mask'.format(value.op.name)
        self._mask = getter(
            name, dtype=tf.bool, shape=shape,
            initializer=tf.ones_initializer(), trainable=False)
        return value * tf.cast(self._mask, tf.float32)

    def _updated_mask(self, var, mask):
        raise NotImplementedError(
            'Method to compute an updated mask is not implemented.')

    def _update(self):
        mask = self._updated_mask(self.before, self._mask)
        return tf.assign(self._mask, mask)


class ThresholdPruner(BasePruner):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def _updated_mask(self, var, mask):
        return _binarize(var, self.threshold)


class MeanStdPruner(BasePruner):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def _threshold(self, tensor):
        axes = list(range(len(tensor.get_shape()) - 1))
        mean, var = tf.nn.moments(tensor, axes=axes)
        return mean + self.alpha * tf.sqrt(var)

    def _updated_mask(self, var, mask):
        return _binarize(var, self._threshold(var))


class DynamicNetworkSurgeryPruner(MeanStdPruner):
    """
    References:
        1. https://github.com/yiwenguo/Dynamic-Network-Surgery
        2. https://arxiv.org/abs/1608.04493
    """
    def __init__(self, c_rate, on_factor=1.1, off_factor=0.9):
        super().__init__(c_rate)
        self.on_factor = on_factor
        self.off_factor = off_factor

    def _updated_mask(self, var, mask):
        threshold = self._threshold(var)
        on_mask = tf.abs(var) > self.on_factor * threshold
        mask = tf.logical_or(mask, on_mask)
        off_mask = tf.abs(var) <= self.off_factor * threshold
        return tf.logical_and(mask, off_mask)


DNSPruner = DynamicNetworkSurgeryPruner


class Rounder(BaseOverrider):
    @staticmethod
    def _apply(getter, value):
        return _round(value)


class FixedPointQuantizer(BaseOverrider):
    """
    Quantize inputs into 2's compliment n-bit fixed-point values with d-bit
    dynamic range.

    Args:
        - width:
            The number of bits to use in number representation.
        - fractional_width:
            The number of bits to use for fractional part.
            If not specified, `.update()` can update the dynamic range of the
            variable automatically.

    References:
        [1] https://arxiv.org/pdf/1604.03168
        [2] https://arxiv.org/pdf/1412.7024
    """
    def __init__(self, width, dynamic_range=None):
        super().__init__()
        self.width = width
        if width < 1:
            raise ValueError(
                'Width of quantized value must be greater than 0.')
        self.dynamic_range = dynamic_range

    def _apply(self, getter, value):
        if not self.dynamic_range:
            name = '{}/dynamic_range'.format(value.op.name)
            self.dynamic_range = getter(
                name, dtype=tf.int32, shape=[],
                initializer=tf.constant_initializer(32),
                trainable=False)

        shift = 2 ** self.dynamic_range
        # x << f - d bits
        value = tf.multiply(value, shift)
        # quantize
        value = Rounder._apply(getter, value)
        # ensure number is representable without overflow
        max_value = 2 ** (self.width - 1)
        value = _clip_by_value(value, -max_value, max_value - 1)
        # revert bit-shift earlier
        value = tf.divide(value, shift)
        return value

    def _update(self):
        raise NotImplementedError
