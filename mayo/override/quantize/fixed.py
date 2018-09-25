from mayo.log import log
from mayo.override import util
from mayo.override.base import Parameter
from mayo.override.quantize.base import QuantizerBase


class ThresholdBinarizer(QuantizerBase):
    threshold = Parameter('threshold', 0, [], 'float')

    def __init__(self, session, threshold=None, should_update=True, enable=True):
        super().__init__(session, should_update, enable)
        if threshold is not None:
            self.threshold = threshold

    def _apply(self, value):
        return util.cast(value > self.threshold, float)


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
    width = Parameter('width', 64, [], 'int')
    point = Parameter('point', 8, [], 'int')

    def __init__(
            self, session, point=None, width=None, stochastic=None,
            should_update=True, enable=True):
        super().__init__(session, should_update, enable)
        if point is not None:
            self.point = point
        if width is not None:
            if width < 1:
                raise ValueError(
                    'Width of quantized value must be greater than 0.')
            self.width = width
        if stochastic is None:
            self.stochastic = False
        self.stochastic = stochastic

    def _quantize(
            self, value, point=None, width=None, compute_overflow_rate=False):
        point = util.cast(self.point if point is None else point, float)
        width = util.cast(self.width if width is None else width, float)
        # x << (width - point)
        shift = 2.0 ** (util.round(width) - util.round(point))
        value = value * shift
        # quantize
        if self.stochastic:
            value = util.stochastic_round(value, self.stochastic)
        else:
            value = util.round(value)
        # ensure number is representable without overflow
        if width is not None:
            max_value = util.cast(2 ** (width - 1), float)
            if compute_overflow_rate:
                overflow_value = value[value != 0]
                overflows = util.logical_or(
                    overflow_value < -max_value,
                    overflow_value > max_value - 1)
                return self._overflow_rate(overflows)
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
    def __init__(
            self, session, width, overflow_rate, stochastic=None,
            should_update=True, enable=True):
        super().__init__(
            session, None, width, stochastic, should_update, enable)
        self.overflow_rate = overflow_rate
        # self.sync_point = sync_point

    def _update_policy(self, tensor):
        raise NotImplementedError

    def _update(self):
        self.point = self._update_policy(self.eval(self.before))

    def search(self, params):
        max_bound = params.get('max')
        if max_bound is None:
            raise ValueError(
                'require max value to search for {}', self.__name__)
        targets = params.get('targets')
        if targets is None or 'point' not in targets:
            raise ValueError(
                'Required targets are not specified')
        w = self.eval(self.width)
        max_value = 2 ** (w - 1)
        for p in range(-2 * w, w + 1):
            shift = 2.0 ** (p)
            if max_bound <= max_value * shift:
                return {'point': w + p}
        log.warn(
            'Cannot find a binary point position that satisfies the '
            'overflow_rate budget, using integer (point at the right '
            'of LSB) instead.')
        return {'point': w}


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
    width = Parameter('width', 16, [], 'float', trainable=True)

    def __init__(self, session, overflow_rate, should_update=True):
        super().__init__(session, None, None, should_update=should_update)

    def _apply(self, value):
        return self._quantize(value, self.width, self.point)


class LogQuantizer(QuantizerBase):
    def __init__(
            self, session, width, overflow_rate, stochastic=None,
            should_update=True, enable=True):
        super().__init__(session, should_update, enable)
        self.width = width
        if width is not None and width < 1:
            raise ValueError(
                'Width of quantized value must be greater than 0.')
        # internal fixed-point quantizer to quantize value in log-domain
        self.quantizer = DGQuantizer(
            session, width, overflow_rate, should_update, stochastic)

    def _quantize(self, value, point, width, compute_overflow_rate=False):
        # decompose
        sign = util.cast(value > 0, float) - util.cast(value < 0, float)
        value = util.log(util.abs(value), 2.0)
        # quantize
        value = self.quantizer.apply(
            value, compute_overflow_rate=compute_overflow_rate)
        if compute_overflow_rate:
            return value
        # represent
        return util.where(util.nonzero(sign), sign * (2 ** value), 0)

    def assign_parameters(self):
        super().assign_parameters()
        self.quantizer.assign_parameters()

    def _update(self):
        self.quantizer.update()

    def _info(self):
        info = self.quantizer.info()._asdict()
        return self._info_tuple(**info)
