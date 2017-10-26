import numpy as np
import tensorflow as tf

from mayo.log import log
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
            The position of the binary point, counting from the LSB.

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

    def _quantize(
            self, value, point=None, width=None, compute_overflow_rate=False):
        if point is None:
            point = self.point
        if width is None:
            width = self.width
        # x << (width - point)
        shift = util.cast(2 ** (width - point), float)
        value = value * shift
        # quantize
        value = util.round(value)
        # ensure number is representable without overflow
        max_value = util.cast(2 ** (width - 1), float)
        if compute_overflow_rate:
            overflows = util.logical_or(
                value < -max_value, value > max_value - 1)
            return _overflow_rate(overflows)
        value = util.clip_by_value(value, -max_value, max_value - 1)
        # revert bit-shift earlier
        return value / shift

    def _apply(self, getter, value):
        return self._quantize(value)

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
        log.warn(
            'Cannot find a binary point position that satisfies the '
            'overflow_rate budget, using integer (point at the right '
            'of LSB) instead.')
        return w


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


class MayoFixedPointQuantizer(FixedPointQuantizer):
    def __init__(self, point, width, should_update=True):
        super().__init__(point, width, should_update)

    def _apply(self, getter, value):
        name = '{}/width'.format(value.op.name)
        self.width_var = getter(
            name, dtype=tf.int32,
            initializer=self.width, trainable=False)
        return self._quantize(value, width=self.width_var, point=self.point)

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

    def _update(self, session):
        tf.assign(self.width_var, self.width)
        return


class MayoDFPQuantizer(DGQuantizer):
    def __init__(self, width, overflow_rate, should_update=True):
        super().__init__(width, overflow_rate, should_update)

    def _threshold_update(self):
        self.width -= self.scale

    def _scale_roll_back(self):
        self.width += self.scale

    def _scale_update(self, update_factor):
        self.scale = round(self.scale * update_factor)

    def _setup(self, session):
        self.scale = round(session.config.retrain.scale)


class FloatingPointQuantizer(QuantizerBase):
    """
    Minifloat quantization.

    When exponent_width is 0, the floating-point value is a degenerate case
    where exponent is always a constant bias, equivalent to fixed-point with a
    sign-magnitude representation.

    When mantissa_width is 0, the floating-point value is a degenerate
    case where mantissa is always 1, equivalent to shifts with only 2^n
    representations.

    When both exponent_width and mantissa_width are 0, the quantized value can
    only represent $2^{-bias}$ or 0, which is not very useful.
    """
    def __init__(
            self, exponent_width, exponent_bias, mantissa_width,
            should_update=True):
        super().__init__(should_update)
        self.exponent_width = exponent_width
        self.exponent_bias = exponent_bias
        self.mantissa_width = mantissa_width
        is_valid = exponent_width >= 0 and mantissa_width >= 0
        is_valid = not (exponent_width == 0 and mantissa_width == 0)
        if not is_valid:
            raise ValueError(
                'We expect exponent_width >= 0 and mantissa_width >= 0 '
                'where equalities must be exclusive.')

    def _decompose(self, value):
        # decompose a single-precision floating-point into
        # sign, exponent and mantissa components
        descriminator = (2 ** self.exponent_bias) / 2
        sign = util.cast(value > descriminator, int)
        sign -= util.cast(value < -descriminator, int)
        value = util.abs(value)
        exponent = util.floor(util.log(value, 2))
        mantissa = value / (2 ** exponent)
        return sign, exponent, mantissa

    def _transform(self, sign, exponent, mantissa):
        # clip exponent and quantize mantissa
        exponent_min = self.exponent_bias
        exponent_max = exponent_min + 2 ** self.exponent_width - 1
        exponent = util.clip_by_value(exponent, exponent_min, exponent_max)
        shift = util.cast(2 ** self.mantissa_width, float)
        mantissa = util.round(mantissa * shift) / shift
        # if the mantissa value gets rounded to >= 2 then we need to divide it
        # by 2 and increment exponent by 1
        is_out_of_range = tf.greater_equal(mantissa, 2)
        mantissa = util.where(is_out_of_range, mantissa / 2, mantissa)
        exponent = util.where(is_out_of_range, exponent + 1, exponent)
        return sign, exponent, mantissa

    def _represent(self, sign, exponent, mantissa):
        # represent the value in floating-point using
        # sign, exponent and mantissa
        if util.is_constant(sign, exponent, mantissa):
            zeros = 0
        elif util.is_numpy(sign, exponent, mantissa):
            zeros = np.zeros(sign.shape, dtype=np.int32)
        else:
            zeros = tf.zeros(sign.shape, dtype=tf.int32)
        value = util.cast(sign, float) * (2.0 ** exponent) * mantissa
        return util.where(
            tf.equal(sign, zeros), util.cast(zeros, float), value)

    def _quantize(self, value):
        sign, exponent, mantissa = self._decompose(value)
        sign, exponent, mantissa = self._transform(sign, exponent, mantissa)
        return self._represent(sign, exponent, mantissa)

    def _apply(self, getter, value):
        quantized = self._quantize(value)
        nan = tf.reduce_sum(tf.cast(tf.is_nan(quantized), tf.int32))
        assertion = tf.Assert(tf.equal(nan, 0), [nan])
        with tf.control_dependencies([assertion]):
            return value + tf.stop_gradient(quantized - value)


class ShiftQuantizer(FloatingPointQuantizer):
    def __init__(
            self, overflow_rate, width=None, bias=None, should_update=True):
        super().__init__(
            exponent_width=width, exponent_bias=bias,
            mantissa_width=0, should_update=should_update)

    def _quantize(self, value):
        sign, exponent, mantissa = self._decompose(value)
        sign, exponent, mantissa = self._transform(sign, exponent, mantissa)
        # mantissa == 1
        return self._represent(sign, exponent, 1)


class LogQuantizer(QuantizerBase):
    def __init__(self, point, width=None, should_update=True):
        super().__init__(should_update)
        self.width = width
        self.point = point
        if width is not None and width < 1:
            raise ValueError(
                'Width of quantized value must be greater than 0.')

    def _decompose(self, value):
        ...

    def _transform(self):
        ...

    def _represent(self):
        ...

    def _quantize(self, value, point, width, compute_overflow_rate=False):
        sign = util.cast(value > 0, float) - util.cast(value < 0, float)
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
        value = 2 ** value * sign
        return value
