import functools
from collections import Sequence, namedtuple

import tensorflow as tf
from tensorflow.python.ops.init_ops import Initializer

from mayo.log import log
from mayo.util import memoize_property, ShapeError


class OverriderError(Exception):
    """All overrider-related exceptions.  """


class OverrideNotAppliedError(OverriderError):
    """Always invoke .apply() before .update().  """


class OverrideAlreadyAppliedError(OverriderError):
    """Do not repeatly invoke .apply().  """


class GetterInvokedOutsideApplyError(OverriderError):
    """Function getter() is invoked not in apply().  """


def _getter_not_initialized(*args, **kwargs):
    raise GetterInvokedOutsideApplyError(
        'The function `getter()` should only be invoked in `.apply()`.')


class Parameter(object):
    """ `tf.Variable`-based overrider hyperparameter.  """
    _dtype_map = {
        'int': tf.int32,
        'float': tf.float32,
        'bool': tf.bool,
    }

    def __init__(
            self, name, initial=None, shape=None,
            dtype='float', trainable=False):
        super().__init__()
        self.name = name
        self.initial = initial
        self.shape = shape
        self.dtype = self._dtype_map[dtype]
        self.trainable = trainable

    def _getter_kwargs(self, instance):
        defaults = instance._parameter_config.get(self.name, {})
        kwargs = {}
        for key in ['initial', 'shape']:
            value = getattr(self, key, None)
            if value is None:
                try:
                    value = defaults[key]
                except KeyError:
                    raise KeyError(
                        'Parameter {} does not specify a configuration for {}.'
                        .format(self.name, key))
            kwargs[key] = value
        kwargs['name'] = self.name
        init = kwargs.pop('initial')
        if init is not None and not isinstance(init, Initializer):
            init = tf.constant_initializer(
                value=init, dtype=self.dtype, verify_shape=True)
        kwargs['initializer'] = init
        kwargs['dtype'] = self.dtype
        kwargs['trainable'] = self.trainable
        return kwargs

    def __get__(self, instance, owner):
        if instance is None:
            return self
        try:
            return instance._parameter_variables[self.name]
        except KeyError:
            pass
        kwargs = self._getter_kwargs(instance)
        var = instance._getter(**kwargs)
        if not isinstance(var, tf.Variable):
            raise TypeError(
                'Curious, variable instantiation does not return a variable.')
        instance._parameter_variables[self.name] = var
        return var

    def __set__(self, instance, value):
        instance._parameter_variables_assignment[self.name] = value


class OverriderBase(object):
    """
    Base class for applying overriding operations on a Net.  Please ensure
    both methods `_apply` and `_update` are overridden with appropriate
    implementations.

    The method `_apply` overrides the variable in `value`, returns the
    overridden result; `_update` updates states of tensorflow variables used in
    `_apply`.
    """
    enable = Parameter('enable', True, [], 'bool')

    def __init__(self, session, enable=True, should_update=True):
        super().__init__()
        self._applied = False
        self.name = None
        self.session = session
        self.internals = {}
        self._parameter_config = {}
        self._parameter_variables = {}
        self._parameter_variables_assignment = {}
        self._getter = _getter_not_initialized
        self.should_update = should_update
        self.enable = enable

    @memoize_property
    def parameters(self):
        params = {}
        for key in dir(self.__class__):
            if key == 'parameters':
                continue
            value = getattr(self.__class__, key)
            if isinstance(value, Parameter):
                params[value.name] = value
        return params

    def assign_parameters(self):
        for name, value in self._parameter_variables_assignment.items():
            if value is None:
                continue
            value_desc = str(value).split('\n')
            multiline = len(value_desc) > 1
            value_desc = value_desc[0] + ('...' if multiline else '')
            log.debug(
                'Assigning overrider parameter: {}.{} = {}'
                .format(self, name, value_desc))
            # ensure variable is instantiated
            self.parameters[name].__get__(self, None)
            # assignment
            var = self._parameter_variables[name]
            try:
                self.session.assign(var, value, raw_run=True)
            except ValueError:
                raise ShapeError(
                    'Variable {!r} in overrider {!r} expects '
                    'its assigned value {!r} to match its shape {!r}.'
                    .format(var, self, value, var.shape))
            # add our variable to the list of initialized_variables
            if var not in self.session.initialized_variables:
                self.session.initialized_variables.append(var)
        self._parameter_variables_assignment = {}

    def _apply(self, value):
        """
        Override this method called in `.apply()` to modify the
        variable in `value`.
        """
        raise NotImplementedError(
            'Overrider method `._apply()` must be implemented.')

    def _tracking_getter(self, getter, scope):
        @functools.wraps(getter)
        def wrapped(name, *args, **kwargs):
            var_name = '{}/{}.{}'.format(scope, self.__class__.__name__, name)
            var = getter(var_name, *args, **kwargs)
            self.internals[name] = var
            return var
        wrapped.__qualname__ = '{}.wrapped.{}'.format(
            self._tracking_getter.__qualname__, getter)
        return wrapped

    def apply(self, node, scope, getter, value):
        """
        Things to apply to the variable in `value`, returns the
        overridden result.
        """
        if self._applied:
            raise OverrideAlreadyAppliedError(
                'Overrider has already been applied to {!r}'.format(value))
        self._applied = True
        self.node = node
        self.name = value.op.name
        self.before = value
        self._scope = scope
        self._original_getter = getter
        self._getter = self._tracking_getter(getter, scope)
        self.overridden = self._apply(value)
        self.after = tf.cond(
            self.enable, lambda: self.overridden, lambda: value)
        # ensure instantiation of all parameter variables
        for param in self.parameters.values():
            param.__get__(self, None)
        return self.after

    def _update(self):
        """
        Override this method called in `.update()` to update internal
        states of the overrider.
        """
        pass

    def update(self):
        """Update things to apply during training.  """
        if not self.should_update:
            return
        if not self._applied:
            raise OverrideNotAppliedError(
                'Method "apply" must be invoked before call "update".')
        name = '{}.update()'.format(self.__class__.__name__)
        with self.session.ensure_graph_unchanged(name):
            self._update()
        log.debug('Updated overrider {!r}.'.format(self.info()))

    def assign(self):
        """Assign overridden values to parameters before overriding.  """
        self.session.assign(self.before, self.after)

    def reset(self):
        """Reset internal variables to their respective initial values.  """
        for var in self.internals.values():
            self.session.assign(var, var.initial_value)

    def _info_tuple(self, **kwargs):
        # relies on dict ordering
        cls = self.__class__.__name__
        cls_name = '{}Info'.format(cls)
        Tuple = namedtuple(cls_name, [cls] + list(kwargs))
        kwargs[cls] = self.name
        return Tuple(**kwargs)

    def _info(self):
        return self._info_tuple()

    def info(self):
        return self._info()

    def estimate(self, layer_info, info):
        """ Override this method to modify layer estimation statistics.  """
        return layer_info

    @classmethod
    def finalize_info(cls, table):
        pass

    def __repr__(self):
        if not self.name:
            return super().__repr__()
        return '<{} overrides {!r}>'.format(
            self.__class__.__qualname__, self.name)

    def quantize_loss(self):
        return tf.reduce_sum(tf.abs(self.after - self.before))


class EmptyOverrider(OverriderBase):
    def _apply(self, value):
        return value


class ChainOverrider(OverriderBase, Sequence):
    """ Composition of overriders.  """
    def __init__(self, session, overriders, should_update=True):
        super().__init__(session, should_update)
        self._check_repetition(overriders)
        self._overriders = overriders

    @staticmethod
    def _check_repetition(overriders):
        cls_names = []
        for o in overriders:
            cls_name = o.__class__.__name__
            if cls_name in cls_names:
                raise TypeError(
                    'We do not support overriding with repeated overrider '
                    'types, {} is defined twice.'.format(cls_name))
            cls_names.append(cls_name)

    def __getitem__(self, index):
        return self._overriders[index]

    def __len__(self):
        return len(self._overriders)

    def assign_parameters(self):
        for o in self._overriders:
            o.assign_parameters()

    def _apply(self, value):
        for o in self._overriders:
            value = o.apply(
                self.node, self._scope, self._original_getter, value)
        return value

    def _update(self):
        for o in self._overriders:
            o.update()

    def reset(self):
        for o in self._overriders:
            o.reset()

    def _info(self):
        return self._info_tuple(overriders=self._overriders)

    def __repr__(self):
        return repr(self._overriders)
