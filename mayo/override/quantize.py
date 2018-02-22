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

    def eval(self, attribute):
        if util.is_tensor(attribute):
            return self.session.run(attribute)
        return attribute


class ThresholdBinarizer(QuantizerBase):
    def __init__(self, session, threshold):
        super().__init__(session)
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

    def __init__(self, session, point=None, width=None, should_update=True):
        super().__init__(session, should_update)
        if point is not None:
            self.point = point
        if width is not None:
            if width < 1:
                raise ValueError(
                    'Width of quantized value must be greater than 0.')
            self.width = width

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
                overflow_value = value[value != 0]
                overflows = util.logical_or(
                    overflow_value < -max_value,
                    overflow_value > max_value - 1)
                return _overflow_rate(overflows)
            value = util.clip_by_value(value, -max_value, max_value - 1)
        # revert bit-shift earlier
        return value / shift

    def _apply(self, value):
        return self._quantize(value)

    def _info(self):
        width = int(self.eval(self.width))
        point = int(self.eval(self.point))
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
    def __init__(self, session, width, overflow_rate, should_update=True):
        super().__init__(session, None, width, should_update=should_update)
        self.overflow_rate = overflow_rate
        # self.sync_point = sync_point

    def _update_policy(self, tensor):
        raise NotImplementedError

    def _update(self):
        self.point = self._update_policy(self.eval(self.before))


class CourbariauxQuantizer(DynamicFixedPointQuantizerBase):
    _initial_point = 1

    def _update_policy(self, tensor):
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
    def _update_policy(self, tensor):
        """ simple brute-force, optimal result.  """
        w = self.eval(self.width)
        for p in range(-w, w + 1):
            rate = self._quantize(
                tensor, point=p, width=w, compute_overflow_rate=True)
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

    def __init__(self, session, overflow_rate, should_update=True):
        super().__init__(session, None, None, should_update=should_update)

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
    width = Parameter('width', 32, [], tf.float32)
    exponent_bias = Parameter('exponent_bias', -127, [], tf.float32)
    mantissa_width = Parameter('mantissa_width', 23, [], tf.float32)

    def __init__(
            self, session, width, exponent_bias, mantissa_width,
            should_update=True):
        super().__init__(session, should_update)
        self.width = width
        self.exponent_bias = exponent_bias
        self.mantissa_width = mantissa_width
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
        """ Clip exponent and quantize mantissa.  """
        exponent_min = -exponent_bias
        exponent_max = 2 ** exponent_width - 1 - exponent_bias
        exponent = util.clip_by_value(exponent, exponent_min, exponent_max)
        shift = util.cast(2 ** mantissa_width, float)
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

    def find_float_exp(self, value, width, overflow_rate):
        """ Compute an exponent bound based on the overflow rate.  """
        max_exponent = int(2 ** width)
        for exp in range(min(-max_exponent, -4), max(max_exponent, 4)):
            max_value = 2 ** (exp + 1)
            overflows = util.logical_or(value < -max_value, value > max_value)
            if _overflow_rate(overflows) <= overflow_rate:
                return exp
        return 0

    def compute_quantization_loss(
            self, value, exponent_width, mantissa_width, overflow_rate):
        max_exponent = self.find_float_exp(
            value, exponent_width, overflow_rate)
        # obtain exponent bias based on the bound
        # max_exponent = exponent - bias, bias >= 0
        exponent_bias = 2 ** exponent_width - 1 - max_exponent
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


class ShiftQuantizer(FloatingPointQuantizer):
    def __init__(
            self, session, overflow_rate, width=None, bias=None,
            should_update=True):
        super().__init__(
            session=session, width=width, exponent_bias=bias,
            mantissa_width=0, should_update=should_update)
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
            if _overflow_rate(overflows) <= self.overflow_rate:
                break
        return exp

    def _update(self):
        log.info('finding a exp bias for shift quantizer using overflow rate')
        max_exponent = self.find_shift_exp(self.eval(self.before))
        self.exponent_bias = 2 ** self.eval(self.width) - 1 - max_exponent


class LogQuantizer(QuantizerBase):
    def __init__(self, session, point, width=None, should_update=True):
        super().__init__(session, should_update)
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
            if instance is None:
                return self
            var = super().__get__(instance, owner)
            return instance._quantize(var, mean_quantizer=True)

    positives = Parameter('positives', None, None, tf.bool)
    positives_mean = QuantizedParameter('positives_mean', 1, [], tf.float32)
    negatives_mean = QuantizedParameter('negatives_mean', -1, [], tf.float32)

    def __init__(
            self, session, quantizer, mean_quantizer=None, should_update=True):
        super().__init__(session, should_update)
        cls, params = object_from_params(quantizer)
        self.quantizer = cls(session, **params)
        self.mean_quantizer = None
        if mean_quantizer:
            cls, params = object_from_params(mean_quantizer)
            self.mean_quantizer = cls(session, **params)

    @memoize_property
    def negatives(self):
        return util.logical_not(self.positives)

    def assign_parameters(self):
        super().assign_parameters()
        self.quantizer.assign_parameters()
        if self.mean_quantizer:
            self.mean_quantizer.assign_parameters()

    def _quantize(self, value, mean_quantizer=False):
        quantizer = self.mean_quantizer if mean_quantizer else self.quantizer
        quantizer = quantizer or self.quantizer
        scope = '{}/{}'.format(self._scope, self.__class__.__name__)
        if mean_quantizer and self.mean_quantizer:
            scope = '{}/mean'.format(scope)
        return quantizer.apply(self.node, scope, self._original_getter, value)

    def _apply(self, value):
        # dynamic parameter configuration
        self._parameter_config = {
            'positives': {
                'initial': tf.ones_initializer(tf.bool),
                'shape': value.shape,
            },
        }
        positives = util.cast(self.positives, float)
        negatives = util.cast(self.negatives, float)
        non_zeros = util.cast(tf.not_equal(self.before, 0), float)

        positives_centralized = positives * (value - self.positives_mean)
        negatives_centralized = negatives * (value - self.negatives_mean)
        quantized = self._quantize(
            non_zeros * (positives_centralized + negatives_centralized))
        value = non_zeros * positives * (quantized + self.positives_mean)
        value += non_zeros * negatives * (quantized + self.negatives_mean)
        return value

    def _update(self):
        # update positives mask and mean values
        value = self.eval(self.before)
        # divide them into two groups
        # mean = util.mean(value)
        mean = 0.0
        # find two central points
        positives = value > mean
        self.positives = positives
        self.positives_mean = util.mean(value[util.where(positives)])
        negatives = util.logical_and(util.logical_not(positives), value != 0)
        self.negatives_mean = util.mean(value[util.where(negatives)])
        # update internal quantizer
        self.quantizer.update()

    def _info(self):
        info = self.quantizer.info()._asdict()
        if self.mean_quantizer:
            mean_info = self.mean_quantizer.info()
            for key, value in mean_info._asdict().items():
                info['mean_' + key] = value
        return self._info_tuple(**info)


class IncrementalQuantizer(OverriderBase):
    """
    https://arxiv.org/pdf/1702.03044.pdf
    """
    interval = Parameter('interval', 0.1, [], tf.float32)
    mask = Parameter('mask', None, None, tf.bool)

    def __init__(self, session, quantizer, intervals, should_update=True):
        super().__init__(session, should_update)
        cls, params = object_from_params(quantizer)
        self.quantizer = cls(session, **params)
        if intervals is not None:
            self.intervals = intervals
            self.interval = intervals[0]
            self.interval_index = 0

    def _quantize(self, value, mean_quantizer=False):
        quantizer = self.quantizer
        scope = '{}/{}'.format(self._scope, self.__class__.__name__)
        return quantizer.apply(self.node, scope, self._original_getter, value)

    def _apply(self, value):
        self._parameter_config = {
            'mask': {
                'initial': tf.zeros_initializer(tf.bool),
                'shape': value.shape,
            }
        }

        quantized_value = self._quantize(value)
        off_mask = util.cast(
            util.logical_not(util.cast(self.mask, bool)), float)
        mask = util.cast(self.mask, float)
        # on mask indicates the quantized values
        return value * off_mask + quantized_value * mask

    def _policy(self, value, quantized, previous_mask, interval):
        previous_pruned = util.sum(previous_mask)
        th_arg = util.cast(util.count(value) * interval, int)
        if th_arg < 0:
            raise ValueError(
                'mask has {} elements, interval is {}'.format(
                    previous_pruned, interval))
        off_mask = util.cast(
            util.logical_not(util.cast(previous_mask, bool)), float)
        metric = value - quantized
        flat_value = metric * off_mask
        flat_value = flat_value.flatten()
        if interval >= 1.0:
            th = util.top_k(util.abs(flat_value), th_arg)
        else:
            th = flat_value.min()
        th = util.cast(th, float)
        new_mask = util.logical_not(util.greater_equal(util.abs(metric), th))
        return util.logical_or(new_mask, previous_mask)

    # override assign_parameters to assign quantizer as well
    def assign_parameters(self):
        super().assign_parameters()
        self.quantizer.assign_parameters()

    def update_interval(self):
        if self.intervals == []:
            return False
        self.session.assign(self.interval, self.intervals[self.interval_index])
        self.interval_index += 1
        return True

    def _update(self):
        # reset index
        self.update_interval()
        self.quantizer.update()
        # if chosen quantized, change it to zeros
        value, quantized, mask, interval = self.session.run(
            [self.before, self.quantizer.after, self.mask, self.interval])
        new_mask = self._policy(value, quantized, mask, interval)
        self.session.assign(self.mask, new_mask)


class MixedPrecisionQuantizer(OverriderBase):
    '''
        Mixed Precision should be implemnted as the following:
        mask1 * precision1 + mask2 * precision2 ...
        The masks are mutually exclusive
        Currently supporting:
            1. making a loss to the reg term
            2. quantizer_maps contains parallel quantizers that each can have
            a different quantizer
            3. channel wise granuarity based on output channels
        TODO:
        provide _update()
    '''
    quantizer_maps = {}
    interval = Parameter('interval', 0.1, [], tf.float32)
    channel_mask = Parameter('channel_mask', None, None, tf.int32)

    def __init__(self, session, quantizers, intervals, picked_quantizer,
                 should_update=True, reg_factor=0.0):
        super().__init__(session, should_update)
        for key, item in dict(quantizers).items():
            cls, params = object_from_params(item)
            quantizer = cls(session, **params)
            self.quantizer_maps[key] = quantizer
        if intervals is not None:
            self.intervals = intervals
            self.interval = intervals[0]
            self.interval_index = 0
        self.reg_factor = reg_factor
        self.picked_quantizer = picked_quantizer

    def _apply(self, value):
        '''
        making an quantization loss to reg loss
        '''
        self._parameter_config = {
            'channel_mask': {
                'initial': tf.zeros_initializer(tf.bool),
                'shape': value.shape,
            }
        }
        quantized_values = self._quantize(value)
        self._quantization_loss(value, quantized_values[self.picked_quantizer])
        # on mask indicates the quantized values
        return self._combine_masks(value, quantized_values)

    def _quantize(self, value, mean_quantizer=False):
        quantized_values = {}
        for key, quantizer in dict(self.quantizer_maps).items():
            scope = '{}/{}'.format(self._scope, self.__class__.__name__ + key)
            quantized_values[key] = quantizer.apply(
                self.node, scope, self._original_getter, value)
        return quantized_values

    def _combine_masks(self, value, quantized_values):
        '''
        Args:
            quantized_value: the current mask is working on this current
                quantized value, this value is not included in
                quantizer_maps
        '''
        if self.quantizer_maps:
            index = 0
            for key, quantizer in self.quantizer_maps.items():
                mask_label = index + 1
                channel_mask = util.cast(
                    util.equal(self.channel_mask, mask_label), float)
                if index == 0:
                    result = quantized_values[key] * channel_mask
                else:
                    result += quantized_values[key] * channel_mask
            # now handel off_mask
        off_mask = util.cast(util.equal(self.channel_mask, 0), float)
        return value * off_mask + result

    def _quantization_loss(self, value, quantized_value):
        loss = tf.reduce_sum(tf.abs(value - quantized_value))
        loss *= self.reg_factor
        loss_name = tf.GraphKeys.REGULARIZATION_LOSSES
        tf.add_to_collection(loss_name, loss)
