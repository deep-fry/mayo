import tensorflow as tf

from mayo.override import util
from mayo.override.base import OverriderBase


def _overflow_rate(mask):
    """
    Compute overflow_rate from a given overflow mask.  Here `mask` is a
    boolean tensor where True and False represent the presence and absence
    of overflow repsectively.
    """
    return util.sum(util.cast(mask, int)) / util.count(mask)


class QuantizerBase(OverriderBase):
    def _quantize(self, value):
        raise NotImplementedError

    def _apply(self, getter, value):
        return self._quantize(value)


class ThresholdBinarizer(QuantizerBase):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def _quantize(self, value):
        return util.binarize(value, self.threshold)


class FixedPointQuantizer(OverriderBase):
    """
    Quantize inputs into 2's compliment n-bit fixed-point values with d-bit
    dynamic range.

    Args:
        - width:
            The number of bits to use in number representation.
            If not specified, we do not limit the range of values.
        - point:
            The position of the binary point, counting from the MSB.

    References:
        [1] https://arxiv.org/pdf/1604.03168
    """
    def __init__(self, point, width=None, should_update=True):
        super().__init__(should_update)
        self.point = point
        self.width = width
        if width is not None and width < 1:
            raise ValueError(
                'Width of quantized value must be greater than 0.')

    def _quantize(self, value, point=None, compute_overflow_rate=False):
        point = point or self.point
        # x << (width - point)
        shift = util.cast(2 ** (self.width - point), float)
        value = value * shift
        # quantize
        value = util.round(value)
        # ensure number is representable without overflow
        max_value = util.cast(2 ** (self.width - 1), float)
        if compute_overflow_rate:
            overflows = util.logical_or(
                value < -max_value, value > max_value - 1)
            return _overflow_rate(overflows)
        value = util.clip_by_value(value, -max_value, max_value - 1)
        # revert bit-shift earlier
        return value / shift

    def _apply(self, getter, value):
        return self._quantize(value, self.width, self.point)

    def info(self, session):
        p = self.point
        if isinstance(p, tf.Variable):
            p = int(session.run(p))
        return self._info_tuple(width=self.width, point=p)


class DynamicFixedPointQuantizerBase(FixedPointQuantizer):
    """
    a base class to quantize inputs into 2's compliment `width`-bit fixed-point
    values with `point`-bit dynamic range.

    Args:
        - width:
            The number of bits to use in number representation.
        - overflow_rate:
            The percentage of tolerable overflow in a certain overridden
            variable.  The method `._update_policy()` should be overridden to
            use this information to compute a corresponding binary point using
            an update policy.
    """
    _init_point = 1

    def __init__(self, width, overflow_rate, should_update=True):
        super().__init__(None, width, should_update=should_update)
        self.overflow_rate = overflow_rate

    def _apply(self, getter, value):
        if not self.point:
            name = '{}/point'.format(value.op.name)
            self.point = getter(
                name, dtype=tf.int32, shape=[],
                initializer=tf.constant_initializer(self._init_point),
                trainable=False)
        return self._quantize(value)

    def _update_policy(self, tensor):
        raise NotImplementedError

    def _update(self, session):
        p = self._update_policy(session.run(self.before))
        session.run(tf.assign(self.point, p))


class CourbariauxQuantizer(DynamicFixedPointQuantizerBase):
    def _update_policy(self, tensor):
        """ algorithm described in: https://arxiv.org/pdf/1412.7024  """
        p = self._init_point
        rate = self._quantize(tensor, point=p, compute_overflow_rate=True)
        if rate > self.overflow_rate:
            p -= 1
        elif 2 * rate <= self.overflow_rate:
            p += 1
        return p


class DGQuantizer(DynamicFixedPointQuantizerBase):
    def _update_policy(self, tensor):
        """ simple brute-force, optimal result.  """
        w = self.width
        for p in range(w + 1):
            rate = self._quantize(tensor, point=p, compute_overflow_rate=True)
            if rate <= self.overflow_rate:
                return p


class DGTrainableQuantizer(DGQuantizer):
    """ Backpropagatable precision.  """
    _init_width = 16

    def __init__(self, overflow_rate, should_update=True):
        super().__init__(None, None, should_update=should_update)

    def width_var(self, getter, name):
        if self.width is not None:
            return self.width
        # trainable width, but no gradients can be felt by it at the moment
        name = '{}/width'.format(name)
        self.width = util.round(getter(
            name, dtype=tf.float32, shape=[],
            initializer=tf.constant_initializer(self._init_width),
            trainable=True))
        return self.width

    def _apply(self, getter, value):
        width = self.width_var(getter, value.op.name)
        point = self.point_var(getter, value.op.name)
        return self._quantize(value, width, point)


class MayoDFPQuantizer(DynamicFixedPointQuantizerBase):
    def __init__(self, width, overflow_rate, should_update=True):
        super().__init__(width, overflow_rate, should_update)

    def _threshold_update(self):
        self.width -= self.scale

    def _scale_roll_back(self):
        self.width += self.scale

    def _scale_update(self, update_factor):
        self.scale = round(self.scale * update_factor)
        if self.point < 1:
            raise ValueError(
                'DFP {}, Bitwidth should be bigger than 1'.format(self.point))

    def _setup(self, session):
        self.scale = round(session.config.retrain.scale)


class FloatingPointQuantizer(OverriderBase):
    """ minifloat quantization. """
    def __init__(
            self, exponent_width=None, exponent_bias=None, width=None,
            should_update=True):
        super().__init__(should_update)
        self.exponent_width = exponent_width
        self.exponent_bias = exponent_bias
        self.width = width
        if width is not None and width < 1:
            raise ValueError(
                'Width of quantized value must be greater than 0.')

    def _decompose(self, value):
        # single-precision floating-point
        #  exponent = _floor(_log())
        pass

    def _quantize(self, value, width, exp_width, bias):
        # 2 ^ (exp_max - bias)
        max_exp = util.cast(2 ** exp_width - 1 - bias, float)
        # 2 ^ (0 - bias)
        min_exp = util.cast(- bias, float)
        frac_width = width - exp_width
        max_value = 2 ** max_exp * 2

        # find a base, clip it
        value_sign = util.cast(
            value < 0, float) * (-1) + util.cast(value > 0, float)
        value = tf.clip_by_value(value, 1e-10, max_value)
        value = tf.abs(value)
        exp_value = util.floor(util.log(value, 2))
        exp_value = util.clip_by_value(exp_value, min_exp, max_exp)

        # find delta, quantize it
        self.delta = value / 2 ** exp_value
        delta = self.delta
        shift = util.cast(2 ** (frac_width), float)
        delta = util.round(delta * shift)
        delta = delta / shift

        value = 2**exp_value * delta
        # keep zeros back
        value = value_sign * value
        return value

    def _apply(self, getter, value):
        return self._quantize(
            value, self.exponent_width, self.exponent_width,
            self.exponent_bias)


class ShiftQuantizer(OverriderBase):
    def __init__(
            self, overflow_rate, width=None, bias=None, should_update=True):
        super().__init__(should_update)
        self.width = width
        self.bias = bias
        if width is not None and width < 1:
            raise ValueError(
                'Width of quantized value must be greater than 0.')

    def _quantize(
            self, value, width, bias, compute_overflow_rate=False):
        max_value = 2 ** width - 1 - bias
        min_value = - 2 ** width - bias
        value_sign = util.cast(value > 0, float) - util.cast(value < 0, float)
        value_zeros = util.cast(value != 0, float)
        value = util.clip_by_value(value, 1e-10, 2**(max_value + 1))
        value = util.cast(value, float)
        value = util.log(value, 2.0)
        value = util.round(value)
        # ensure number is representable without overflow
        if compute_overflow_rate:
            overflows = util.logical_or(value < min_value, value > max_value)
            return _overflow_rate(overflows)
        value = util.clip_by_value(value, min_value, max_value)
        value = 2 ** (value) * value_sign * value_zeros
        return value

    def _update(self, session):
        pass
        # self._quantize((self.before), self.width, self.bias)
        return

    def _apply(self, getter, value):
        return self._quantize(value, self.width, self.bias)


class LogQuantizer(OverriderBase):
    def __init__(self, point, width=None, should_update=True):
        super().__init__(should_update)
        self.width = width
        self.point = point
        if width is not None and width < 1:
            raise ValueError(
                'Width of quantized value must be greater than 0.')

    def _quantize(self, value, point, width, compute_overflow_rate=False):
        # fetch signs and zeros
        value_sign = util.cast(value > 0, float) - util.cast(value < 0, float)
        # log only handels positive values
        max_range = 2**(width - point)
        value = util.clip_by_value(1e-10, 2**(max_range))
        value = util.log(util.abs(value), 2.0)
        # quantize to log-domain
        shift = util.cast(2 ** (width - point), float)
        value = shift * value
        value = util.round(value)
        min_value = - 2 ** width
        max_value = 2 ** width - 1
        # ensure number is representable without overflow
        if compute_overflow_rate:
            overflows = util.logical_or(value < min_value, value > max_value)
            return util.sum(util.cast(overflows, int)) / util.count(overflows)
        value = util.clip_by_value(value, min_value, max_value)
        value = value / shift
        value = 2 ** value * value_sign
        return value

    def _update(self, session):
        self._quantize(session.run(self.before), self.width, self.point)
        return

    def _apply(self, getter, value):
        return self._quantize(value, self.width, self.point)
