import numpy as np
import tensorflow as tf

from mayo.log import log
from mayo.util import memoize_property, object_from_params
from mayo.override import util
from mayo.override.base import OverriderBase, Parameter


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

    def _apply(self, value):
        return self._quantize(value)


class ThresholdBinarizer(QuantizerBase):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def _quantize(self, value):
        return util.binarize(value, self.threshold)


class FixedPointQuantizer(QuantizerBase):
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
    width = Parameter('width', 32, [], tf.int32)
    point = Parameter('point', 2, [], tf.int32)

    def __init__(self, point=None, width=None, should_update=True):
        super().__init__(should_update)
        if point is not None:
            self.point = point
        if width is not None:
            if width < 1:
                raise ValueError(
                    'Width of quantized value must be greater than 0.')
            self.width = width

    def eval(self, session, attribute):
        if util.is_tensor(attribute):
            return session.run(attribute)
        return attribute

    def _quantize(
            self, value, point=None, width=None, compute_overflow_rate=False):
        point = util.cast(self.point if point is None else point, float)
        width = util.cast(self.width if width is None else width, float)
        # x << (width - point)
        shift = 2.0 ** (util.round(width) - util.round(point))
        value = value * shift
        # quantize
        value = util.round(value)
        # ensure number is representable without overflow
        if width is not None:
            max_value = util.cast(2 ** (width - 1), float)
            if compute_overflow_rate:
                overflows = util.logical_or(
                    value < -max_value, value > max_value - 1)
                return _overflow_rate(overflows)
            value = util.clip_by_value(value, -max_value, max_value - 1)
        # revert bit-shift earlier
        return value / shift

    def _apply(self, value):
        return self._quantize(value)

    def _info(self, session):
        width = int(self.eval(session, self.width))
        point = int(self.eval(session, self.point))
        return self._info_tuple(width=width, point=point)


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
    def __init__(self, width, overflow_rate, should_update=True):
        super().__init__(None, width, should_update=should_update)
        self.overflow_rate = overflow_rate

    def _update_policy(self, tensor, session):
        raise NotImplementedError

    def _update(self, session):
        self.point = self._update_policy(session.run(self.before), session)


class CourbariauxQuantizer(DynamicFixedPointQuantizerBase):
    _initial_point = 1

    def _update_policy(self, tensor, session):
        """ algorithm described in: https://arxiv.org/pdf/1412.7024  """
        w = self.eval(self.width)
        p = self._initial_point
        rate = self._quantize(
            tensor, width=w, point=p, compute_overflow_rate=True)
        if rate > self.overflow_rate:
            p -= 1
        elif 2 * rate <= self.overflow_rate:
            p += 1
        return p


class DGQuantizer(DynamicFixedPointQuantizerBase):
    def _update_policy(self, tensor, session):
        """ simple brute-force, optimal result.  """
        w = session.run(self.width)
        for p in range(w + 1):
            rate = self._quantize(tensor, point=p, compute_overflow_rate=True)
            rate = session.run(rate)
            if rate <= self.overflow_rate:
                return p
        log.warn(
            'Cannot find a binary point position that satisfies the '
            'overflow_rate budget, using integer (point at the right '
            'of LSB) instead.')
        return w


class DGTrainableQuantizer(DGQuantizer):
    """
    Backpropagatable precision.

    Trainable width, but no gradients can be felt by it at the moment.
    """
    width = Parameter('width', 16, [], tf.float32, trainable=True)

    def __init__(self, overflow_rate, should_update=True):
        super().__init__(None, None, should_update=should_update)

    def _apply(self, value):
        return self._quantize(value, self.width, self.point)


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
    exponent_width = Parameter('exponent_width', 8, [], tf.int32)
    exponent_bias = Parameter('exponent_bias', -127, [], tf.int32)
    mantissa_width = Parameter('mantissa_width', 23, [], tf.int32)

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
        """
        Decompose a single-precision floating-point into
        sign, exponent and mantissa components.
        """
        # smallest non-zero floating point
        descriminator = (2 ** self.exponent_bias) / 2
        sign = util.cast(value > descriminator, int)
        sign -= util.cast(value < -descriminator, int)
        value = util.abs(value)
        exponent = util.floor(util.log(value, 2))
        mantissa = value / (2 ** exponent)
        return sign, exponent, mantissa

    def _transform(self, sign, exponent, mantissa):
        """ Clip exponent and quantize mantissa.  """
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
        """
        Represent the value in floating-point using
        sign, exponent and mantissa.
        """
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

    def _apply(self, value):
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


class Recentralizer(OverriderBase):
    """ Recentralizes the distribution of pruned weights.  """

    class QuantizedParameter(Parameter):
        def __get__(self, instance, owner):
            var = super().__get__(instance, owner)
            return instance._quantize(var)

    positives = Parameter('positives', None, None, tf.bool)
    positives_mean = QuantizedParameter('positives_mean', 1, [], tf.float32)
    negatives_mean = QuantizedParameter('negatives_mean', -1, [], tf.float32)

    def __init__(self, quantizer=None, should_update=True):
        super().__init__(should_update)
        quantizer_cls, params = object_from_params(quantizer)
        self.quantizer = quantizer_cls(**params)

    @memoize_property
    def negatives(self):
        return util.logical_not(self.positives)

    def _quantize(self, value):
        return self.quantizer.apply(self._scope, self._original_getter, value)

    def _apply(self, value):
        self._parameter_config = {
            'positives': {
                'initial': tf.ones_initializer(tf.bool),
                'shape': value.shape,
            },
        }
        positives = util.cast(self.positives, float)
        negatives = util.cast(self.negatives, float)
        positives_centralized = positives * (value - self.positives_mean)
        negatives_centralized = negatives * (value - self.negatives_mean)
        quantized = self._quantize(
            positives_centralized + negatives_centralized)
        value = positives * (quantized + self.positives_mean)
        value += negatives * (quantized + self.negatives_mean)
        return value

    def _update(self, session):
        # update internal quantizer
        self.quantizer.update(session)
        # update positives mask and mean values
        value = session.run(self.before)
        # divide them into two groups
        mean = util.mean(value)
        # find two central points
        positives = value >= mean
        self.positives = positives
        self.positives_mean = util.mean(value[util.where(positives)])
        negatives = util.logical_not(positives)
        self.negatives_mean = util.mean(value[util.where(negatives)])

    def _info(self, session):
        info = self.quantizer.info(session)._asdict()
        return self._info_tuple(**info)
