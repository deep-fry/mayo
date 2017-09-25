from collections import Sequence, namedtuple
from functools import partial

import numpy as np
import tensorflow as tf

from mayo.util import format_percent
from mayo.util import log


def _is_constant(*args):
    return all(isinstance(a, (bool, int, float)) for a in args)


def _is_numpy(*args):
    if _is_constant(*args):
        return False
    return all(isinstance(a, (bool, int, float, np.ndarray)) for a in args)


def _is_tensor(*args):
    return any(isinstance(a, tf.Tensor) for a in args)


def _cast(value, dtype):
    if _is_constant(value):
        return dtype(value)
    if _is_numpy(value):
        dtypes = {
            float: np.float32,
            int: np.int32,
        }
        return np.cast[dtypes[dtype]](value)
    dtypes = {
        float: tf.float32,
        int: tf.int32,
    }
    return tf.cast(value, dtypes[dtype])


def _round(value):
    if _is_constant(value):
        return round(value)
    omap = {'Round': 'Identity'}
    with tf.get_default_graph().gradient_override_map(omap):
        return tf.round(value)


def _abs(value):
    if _is_constant(value):
        return abs(value)
    if _is_numpy(value):
        return np.abs(value)
    return tf.abs(value)


def _binary_bool_operation(a, b, op):
    if _is_constant(a, b):
        raise TypeError('Element-wise operator not supported on scalars.')
    if _is_numpy(a, b):
        return getattr(np, op)(a, b)
    return getattr(tf, op)(a, b)


_logical_or = partial(_binary_bool_operation, op='logical_or')
_logical_and = partial(_binary_bool_operation, op='logical_and')


def _clip(*args, min_max=None):
    if _is_constant(*args):
        return min(*args) if min_max else max(*args)
    if _is_numpy(*args):
        return np.min(*args) if min_max else np.max(*args)
    return tf.minimum(*args) if min_max else tf.maximum(*args)


_min = partial(_clip, min_max=True)
_max = partial(_clip, min_max=False)


def _binarize(tensor, threshold):
    return _cast(_abs(tensor) > threshold, float)


def _clip_by_value(tensor, minimum, maximum, transparent_backprop=False):
    if _is_tensor(tensor, minimum, maximum):
        return _min(_max(tensor, minimum), maximum)
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
    def __init__(self, should_update=True):
        super().__init__()
        self.name = None
        self.should_update = should_update

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
        self.name = value.op.name
        self.before = value
        self.after = self._apply(getter, value)
        return self.after

    def _update(self, session):
        """Update things to apply during training.  """
        pass

    def update(self, session):
        if not self.should_update:
            return None
        if not getattr(self, '_applied', False):
            raise self.__class__.OverrideNotAppliedError(
                'Method "apply" must be invoked before call "update".')
        self._update(session)
        log.debug('Updated overrider {!r}'.format(self.info(session)))

    def info(self, session):
        return self._info(session)

    def _info_tuple(self, **kwargs):
        # relies on dict ordering
        cls = self.__class__.__name__
        cls_name = '{}Info'.format(cls)
        Tuple = namedtuple(cls_name, [cls] + list(kwargs))
        kwargs[cls] = self.name
        return Tuple(**kwargs)

    def _info(self, session):
        return self._info_tuple()

    def __repr__(self):
        if not self.name:
            return super().__repr__()
        return '<{} overrides {!r}>'.format(
            self.__class__.__qualname__, self.name)


class ChainOverrider(BaseOverrider, Sequence):
    def __init__(self, overriders, should_update=True):
        super().__init__(should_update)
        self._overriders = overriders

    def __getitem__(self, index):
        return self._overriders[index]

    def __len__(self):
        return len(self._overriders)

    def _apply(self, getter, value):
        for o in self._overriders:
            value = o.apply(getter, value)
        return value

    def _update(self, session):
        for o in self._overriders:
            o.update(session)

    def _info(self, session):
        return self._info_tuple(overriders=self._overriders)

    def __repr__(self):
        return repr(self._overriders)


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
        mask = _cast(self._mask, float)
        return value * mask

    def _updated_mask(self, var, mask):
        raise NotImplementedError(
            'Method to compute an updated mask is not implemented.')

    def _update(self, session):
        mask = self._updated_mask(self.before, self._mask)
        return session.run(tf.assign(self._mask, mask))

    def _info(self, session):
        mask = session.run(self._mask).astype(int)
        density = np.sum(mask) / mask.size
        density = format_percent(density)
        return self._info_tuple(
            mask=self._mask.name, density=density, count=mask.size)


class ThresholdPruner(BasePruner):
    def __init__(self, threshold, should_update=True):
        super().__init__(should_update)
        self.threshold = threshold

    def _updated_mask(self, var, mask):
        return _binarize(var, self.threshold)


class MeanStdPruner(BasePruner):
    def __init__(self, alpha, should_update=True):
        super().__init__(should_update)
        self.alpha = alpha

    def _threshold(self, tensor):
        # axes = list(range(len(tensor.get_shape()) - 1))
        axes = list(range(len(tensor.get_shape())))
        mean, var = tf.nn.moments(_abs(tensor), axes=axes)
        return mean + self.alpha * tf.sqrt(var)

    def _updated_mask(self, var, mask):
        return _binarize(var, self._threshold(var))


class DynamicNetworkSurgeryPruner(MeanStdPruner):
    """
    References:
        1. https://github.com/yiwenguo/Dynamic-Network-Surgery
        2. https://arxiv.org/abs/1608.04493
    """
    def __init__(
            self, c_rate, on_factor=1.1, off_factor=0.9, should_update=True):
        super().__init__(c_rate, should_update)
        self.on_factor = on_factor
        self.off_factor = off_factor

    def _updated_mask(self, var, mask):
        threshold = self._threshold(var)
        on_mask = _abs(var) > self.on_factor * threshold
        mask = _logical_or(mask, on_mask)
        off_mask = _abs(var) <= self.off_factor * threshold
        return _logical_and(mask, off_mask)


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
            If not specified, we do not limit the range of values.
        - fractional_width:
            The number of bits to use for fractional part.
            If not specified, `.update()` can update the dynamic range of the
            variable automatically.

    References:
        [1] https://arxiv.org/pdf/1604.03168
        [2] https://arxiv.org/pdf/1412.7024
    """
    _full_precision = 32

    def __init__(self, width=None, dynamic_range=None, should_update=True):
        super().__init__(should_update)
        self.width = width
        if width is not None and width < 1:
            raise ValueError(
                'Width of quantized value must be greater than 0.')
        self.dynamic_range = dynamic_range

    def _apply(self, getter, value):
        if self.dynamic_range is None:
            name = '{}/dynamic_range'.format(value.op.name)
            self.dynamic_range = getter(
                name, dtype=tf.int32, shape=[],
                initializer=tf.constant_initializer(self._full_precision),
                trainable=False)

        shift = _cast(2 ** self.dynamic_range, float)
        # x << dynamic_range
        value = value * shift
        # quantize
        value = _round(value)
        # ensure number is representable without overflow
        max_value = _cast(2 ** (self.width - 1), float)
        value = _clip_by_value(value, -max_value, max_value - 1)
        # revert bit-shift earlier
        return value / shift

    def _update_policy(self, tensor):
        raise NotImplementedError

    def _update(self, session):
        if not isinstance(self.dynamic_range, tf.Variable):
            return
        dr = self._update_policy(session.run(self.before))
        session.run(tf.assign(self.dynamic_range, dr))

    def _info(self, session):
        dr = self.dynamic_range
        if isinstance(dr, tf.Variable):
            dr = session.run(dr)
        return self._info_tuple(width=self.width, dynamic_range=dr)


class DynamicFixedPointQuantizer(FixedPointQuantizer):
    """
    Update policy uses https://arxiv.org/pdf/1412.7024.pdf
    """
    def __init__(self, width, overflow_rate, should_update=True):
        super().__init__(
            width, dynamic_range=None, should_update=should_update)

    def _update_policy(self, tensor):
        ...
