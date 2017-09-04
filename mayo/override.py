import tensorflow as tf
import numpy as np


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
        if getattr(self, '_applied', False):
            raise self.__class__.OverrideNotAppliedError(
                '"apply" must be invoked before call "update".')
        return self._update()


class BasePruner(BaseOverrider):
    def _apply(self, getter, value):
        shape = value.get_shape()
        name = '{}/mask'.format(value.op.name)
        ones = tf.constant(np.ones(shape))
        self._mask = getter(
            name, shape=shape, dtype=tf.int32,
            initialier=ones, trainable=False)
        return value * self._mask

    def _updated_mask(self):
        raise NotImplementedError(
            'Method to compute an updated mask is not implemented.')

    def _update(self):
        return tf.assign(self._mask, self._updated_mask())


class ThresholdPruner(BasePruner):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def _updated_mask(self):
        mask = np.absolute(self._before) > self.threshold
        mask.astype(int)
        return mask


class MeanStdThresholdPruner(ThresholdPruner):
    def __init__(self, alpha):
        super.__init__()
        self.alpha = alpha

    def _updated_mask(self):
        threshold = np.mean(self._before) + self.alpha * np.std(self._before)
        return super().threshold(self._before, threshold)


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
        # >> f
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
