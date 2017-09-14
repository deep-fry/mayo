import tensorflow as tf


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
        self._before = value
        self._after = self._apply(getter, value)
        return self._after

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


class ChainOverrider(BaseOverrider):
    def __init__(self, overriders):
        super().__init__()
        self._overriders = overriders

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
        return tf.cast(tf.abs(value) > self.threshold, tf.float32)


class BasePruner(BaseOverrider):
    def _apply(self, getter, value):
        shape = value.get_shape()
        name = '{}/mask'.format(value.op.name)
        self._mask = getter(
            name, dtype=tf.float32, shape=shape,
            initializer=tf.ones_initializer(), trainable=False)
        return value * self._mask

    def _updated_mask(self):
        raise NotImplementedError(
            'Method to compute an updated mask is not implemented.')

    def _update(self):
        return tf.assign(self._mask, self._updated_mask())


class DynamicNetworkSurgery(BasePruner):
    """
    Ref:
    1. https://github.com/yiwenguo/Dynamic-Network-Surgery
    2. https://arxiv.org/abs/1608.04493
    """
    def __init__(self, cRate):
        super().__init__()
        self.cRate = cRate

    def _updated_mask(self):
        axis = list(range(len(self._before.get_shape()) - 1))
        mean, var = tf.nn.moments(self._before, axis)
        std = tf.sqrt(var)
        off_threshold = 0.9 * (mean + self.cRate * std)
        on_threshold = 1.1 * (mean + self.cRate * std)
        # off mask indicates variables that should stay
        off_mask = tf.abs(self._before) > off_threshold
        on_mask = tf.abs(self._before) > on_threshold
        mask = tf.cast(self._mask, tf.bool)
        mask = tf.logical_or(mask, on_mask)
        mask = tf.logical_and(mask, off_mask)
        mask = tf.cast(mask, tf.float32)
        return mask


class ThresholdPruner(BasePruner):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        self._binarizer = ThresholdBinarizer(threshold)

    def _updated_mask(self):
        return self._binarizer.apply(None, self._before)


class MeanStdPruner(BasePruner):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def _updated_mask(self):
        axes = list(range(len(self._before.get_shape())))
        mean, var = tf.nn.moments(self._before, axes=axes)
        threshold = mean + self.alpha * tf.sqrt(var)
        return ThresholdBinarizer(threshold).apply(None, self._before)


class Rounder(BaseOverrider):
    @staticmethod
    def _apply(getter, value):
        omap = {'Round': 'Identity'}
        with tf.get_default_graph().gradient_override_map(omap):
            return tf.round(value)


class DynamicFixedPointQuantizer(BaseOverrider):
    """
    Quantize inputs into 2's compliment fixed-point values with an n-bit
    integer part and an f-bit fractional part with d-bit dynamic range.

    Args:
        - integer_width:
            the number of bits to use in integer part.  If not specified
            (None), then we do not restrict the value bound.
        - fractional_width:
            the number of bits to use in fractional part.
        - dynamic_range:
            the dynamic range to use.

    References:
        - https://arxiv.org/pdf/1604.03168
    """
    def __init__(
            self, integer_width=None, fractional_width=8, dynamic_range=0):
        super().__init__()
        self.integer_width = integer_width
        self.fractional_width = fractional_width
        self.dynamic_range = dynamic_range
        self._rounder = Rounder()

    def _apply(self, getter, value):
        dr = self.dynamic_range
        fw = self.fractional_width
        iw = self.integer_width
        # x = f - d bits
        value *= 2 ** (fw - dr)
        # quantize
        value = self._rounder.apply(getter, value)
        value = tf.div(value, 2 ** fw)
        # ensure number is representable without overflow
        if iw is not None:
            max_value = 2 ** (iw - dr)
            value = tf.clip_by_value(value, -max_value, max_value - 1)
        # restore shift by dynamic range
        if dr != 0:
            value *= 2 ** dr
        return value


class FixedPointQuantizer(DynamicFixedPointQuantizer):
    def __init__(self, integer_width=None, fractional_width=8):
        super().__init__(integer_width, fractional_width, 0)
