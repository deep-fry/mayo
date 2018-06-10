import numpy as np
import tensorflow as tf

from mayo.override import util
from mayo.override.base import Parameter
from mayo.override.quantize.base import QuantizerBase
from mayo.log import log


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
    width = Parameter('width', 32, [], 'float')
    exponent_bias = Parameter('exponent_bias', -127, [], 'float')
    mantissa_width = Parameter('mantissa_width', 23, [], 'float')

    def __init__(
            self, session, width, exponent_bias, mantissa_width,
            overflow_rate=0.0, should_update=True, stochastic=None):
        super().__init__(session, should_update)
        self.width = width
        self.exponent_bias = exponent_bias
        self.mantissa_width = mantissa_width
        self.overflow_rate = overflow_rate
        self.stochastic = stochastic
        exponent_width = width - mantissa_width
        is_valid = exponent_width >= 0 and mantissa_width >= 0
        is_valid = is_valid and (
            not (exponent_width == 0 and mantissa_width == 0))
        if not is_valid:
            raise ValueError(
                'We expect exponent_width >= 0 and mantissa_width >= 0 '
                'where equalities must be exclusive.')

    def _decompose(self, value, exponent_bias=None):
        """
        Decompose a single-precision floating-point into
        sign, exponent and mantissa components.
        """
        if exponent_bias is None:
            exponent_bias = self.exponent_bias
        # smallest non-zero floating point
        descriminator = (2 ** (-exponent_bias)) / 2
        sign = util.cast(value > descriminator, int)
        sign -= util.cast(value < -descriminator, int)
        value = util.abs(value)
        exponent = util.floor(util.log(value, 2))
        mantissa = value / (2 ** exponent)
        return sign, exponent, mantissa

    def _transform(
            self, sign, exponent, mantissa, exponent_width=None,
            mantissa_width=None, exponent_bias=None):
        if exponent_bias is None:
            exponent_bias = self.exponent_bias
        if exponent_width is None:
            exponent_width = self.width - self.mantissa_width
        if mantissa_width is None:
            mantissa_width = self.mantissa_width
        # clip exponent and quantize mantissa
        exponent_min = -exponent_bias
        exponent_max = 2 ** exponent_width - 1 - exponent_bias
        exponent = util.clip_by_value(exponent, exponent_min, exponent_max)
        shift = util.cast(2 ** mantissa_width, float)
        # quantize
        if self.stochastic:
            mantissa = util.stochastic_round(mantissa * shift, self.stochastic)
            mantissa /= shift
        else:
            mantissa = util.round(mantissa * shift) / shift

        # if the mantissa value gets rounded to >= 2 then we need to divide it
        # by 2 and increment exponent by 1
        is_out_of_range = util.greater_equal(mantissa, 2)
        mantissa = util.where(is_out_of_range, mantissa / 2, mantissa)
        exponent = util.where(is_out_of_range, exponent + 1, exponent)
        return sign, exponent, mantissa

    def _represent(self, sign, exponent, mantissa):
        """
        Represent the value in floating-point using
        sign, exponent and mantissa.
        """
        value = util.cast(sign, float) * (2.0 ** exponent) * mantissa
        if util.is_constant(sign, exponent, mantissa):
            return value
        if util.is_numpy(sign, exponent, mantissa):
            zeros = np.zeros(sign.shape, dtype=np.int32)
        else:
            zeros = tf.zeros(sign.shape, dtype=tf.int32)
        is_zero = util.equal(sign, zeros)
        return util.where(is_zero, util.cast(zeros, float), value)

    def _quantize(
            self, value, exponent_width=None, mantissa_width=None,
            exponent_bias=None):
        sign, exponent, mantissa = self._decompose(value, exponent_bias)
        sign, exponent, mantissa = self._transform(
            sign, exponent, mantissa,
            exponent_width, mantissa_width, exponent_bias)
        return self._represent(sign, exponent, mantissa)

    def _apply(self, value):
        quantized = self._quantize(value)
        nan = tf.reduce_sum(tf.cast(tf.is_nan(quantized), tf.int32))
        assertion = tf.Assert(tf.equal(nan, 0), [nan])
        with tf.control_dependencies([assertion]):
            return value + tf.stop_gradient(quantized - value)

    def _bias(self, value, exponent_width, profiled_max=None):
        max_exponent = int(2 ** exponent_width)
        for exponent in range(min(-max_exponent, -4), max(max_exponent, 4)):
            max_value = 2 ** (exponent + 1)
            if profiled_max is not None:
                if profiled_max < max_value:
                    return 2 ** exponent_width - 1 - exponent
            else:
                overflows = util.logical_or(
                    value < -max_value, value > max_value)
                if self._overflow_rate(overflows) <= self.overflow_rate:
                    break
        return 2 ** exponent_width - 1 - exponent

    def compute_quantization_loss(
            self, value, exponent_width, mantissa_width, overflow_rate,
            profiled_max=None):
        exponent_bias = self._bias(value, exponent_width, profiled_max)
        quantized = self._quantize(
            value, exponent_width, mantissa_width, exponent_bias)
        # mean squared loss
        loss = ((value - quantized) ** 2).mean()
        return (loss, exponent_bias)

    def _info(self):
        width = int(self.eval(self.width))
        mantissa_width = int(self.eval(self.mantissa_width))
        exponent_bias = int(self.eval(self.exponent_bias))
        return self._info_tuple(
            width=width, mantissa_width=mantissa_width,
            exponent_bias=exponent_bias)

    def _update(self):
        value = self.eval(self.before)
        exponent_width = self.eval(self.width) - self.eval(self.mantissa_width)
        self.exponent_bias = self._bias(value, exponent_width)

    def search(self, params):
        max_bound = params.get('max')
        if max_bound is None:
            raise ValueError(
                'require max value to search for {}', self.__name__)
        samples = params.get('samples')
        if samples is None:
            raise ValueError(
                'require max value to search for {}', self.__name__)
        targets = params.get('targets')
        if targets is None or 'mantissa_width' not in targets or \
                'exponent_bias' not in targets:
            raise ValueError(
                'Required targets are not specified')
        w = int(self.eval(self.width))
        loss_meta = []
        for mantissa in range(w + 1):
            exp = w - mantissa
            loss, bias = self.compute_quantization_loss(
                samples.flatten(), mantissa, exp, 0, max_bound)
            loss_meta.append([loss, [exp, mantissa, bias]])
        loss_meta.sort(key=lambda x: x[0])
        # pick the one that has smallest quantization loss
        exp, mantissa, bias = loss_meta[0][1]
        selected_targets = {
            'mantissa_width': mantissa,
            'exponent_bias': bias,
        }
        return selected_targets


class ShiftQuantizer(FloatingPointQuantizer):
    def __init__(
            self, session, overflow_rate, width=None, bias=None,
            should_update=True, stochastic=None):
        super().__init__(
            session=session, width=width, exponent_bias=bias,
            mantissa_width=0, should_update=should_update,
            stochastic=stochastic)
        self.overflow_rate = overflow_rate

    def _quantize(self, value):
        sign, exponent, mantissa = self._decompose(value)
        sign, exponent, mantissa = self._transform(sign, exponent, mantissa)
        # mantissa == 1
        return self._represent(sign, exponent, 1)

    def find_shift_exp(self, value):
        width = self.eval(self.width)
        max_exponent = int(2 ** width)
        for exp in range(min(-max_exponent, -4), max(max_exponent, 4)):
            max_value = 2 ** (exp + 1)
            overflows = util.logical_or(value < -max_value, value > max_value)
            if self._overflow_rate(overflows) <= self.overflow_rate:
                break
        return exp

    def _update(self):
        max_exponent = self.find_shift_exp(self.eval(self.before))
        self.exponent_bias = 2 ** self.eval(self.width) - 1 - max_exponent
