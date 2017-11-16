import functools
from collections import Sequence, namedtuple

import tensorflow as tf
from tensorflow.python.ops.init_ops import Initializer

from mayo.log import log
from mayo.util import memoize_property


class OverrideNotAppliedError(Exception):
    """Invoke apply before update.  """


class GetterInvokedOutsideApplyError(Exception):
    """Function getter() is invoked not in apply()."""


def _getter_not_initialized(*args, **kwargs):
    raise GetterInvokedOutsideApplyError(
        'The function `getter()` should only be invoked in `.apply()`.')


class Parameter(object):
    """ `tf.Variable`-based overrider hyperparameter.  """
    def __init__(
            self, name, initial=None, shape=None,
            dtype=tf.float32, trainable=False):
        super().__init__()
        self.name = name
        self.initial = initial
        self.shape = shape
        self.dtype = dtype
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
        try:
            return instance._parameter_variables[self.name]
        except KeyError:
            pass
        kwargs = self._getter_kwargs(instance)
        var = instance._getter(**kwargs)
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

    def __init__(self, should_update=True):
        super().__init__()
        self._applied = False
        self.name = None
        self.internals = {}
        self._parameter_config = {}
        self._parameter_variables = {}
        self._parameter_variables_assignment = {}
        self.should_update = should_update

    @memoize_property
    def parameters(self):
        params = {}
        for key in dir(self):
            if key == 'parameters':
                continue
            value = getattr(self, key)
            if isinstance(value, Parameter):
                params[key] = value
        return params

    def assign_parameters(self, session):
        ops = []
        for name, value in self._parameter_variables_assignment.items():
            if value is None:
                continue
            log.debug(
                'Assigning overrider parameter: {}.{} = {}'
                .format(self, name, value))
            getattr(self, name)  # ensure variable is instantiated
            var = self._parameter_variables[name]
            ops.append(tf.assign(var, value))
            # add our variable to the list of initialized_variables
            if var not in session.initialized_variables:
                session.initialized_variables.append(var)
        session.raw_run(ops)
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
        return wrapped

    def apply(self, scope, getter, value):
        """
        Things to apply to the variable in `value`, returns the
        overridden result.
        """
        self._applied = True
        self.name = value.op.name
        self.before = value
        self._scope = scope
        self._original_getter = getter
        self._getter = self._tracking_getter(getter, scope)
        self.after = self._apply(value)
        # ensure instantiation of all parameter variables
        for param in self.parameters:
            getattr(self, param)
        return self.after

    def _update(self, session):
        """
        Override this method called in `.update()` to update internal
        states of the overrider.
        """
        pass

    def update(self, session):
        """Update things to apply during training.  """
        if not self.should_update:
            return
        if not self._applied:
            raise OverrideNotAppliedError(
                'Method "apply" must be invoked before call "update".')
        with session.as_default():
            self._update(session)
        log.debug('Updated overrider {!r}'.format(self.info(session)))

    def assign(self, session):
        """Assign overridden values to parameters before overriding.  """
        with session.as_default():
            session.run(tf.assign(self.before, self.after))

    def reset(self, session):
        """Reset internal variables to their respective initial values.  """
        with session.as_default():
            for var in self.internals.values():
                session.run(tf.assign(var, var.initial_value))

    def _info_tuple(self, **kwargs):
        # relies on dict ordering
        cls = self.__class__.__name__
        cls_name = '{}Info'.format(cls)
        Tuple = namedtuple(cls_name, [cls] + list(kwargs))
        kwargs[cls] = self.name
        return Tuple(**kwargs)

    def _info(self, session):
        return self._info_tuple()

    def info(self, session):
        with session.as_default():
            return self._info(session)

    @classmethod
    def finalize_info(cls, table):
        pass

    def __repr__(self):
        if not self.name:
            return super().__repr__()
        return '<{} overrides {!r}>'.format(
            self.__class__.__qualname__, self.name)


class EmptyOverrider(OverriderBase):
    def _apply(self, value):
        return value


class ChainOverrider(OverriderBase, Sequence):
    """ Composition of overriders.  """
    def __init__(self, overriders, should_update=True):
        super().__init__(should_update)
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

    def assign_parameters(self, session):
        for o in self._overriders:
            o.assign_parameters(session)

    def _apply(self, value):
        for o in self._overriders:
            value = o.apply(self._scope, self._original_getter, value)
        return value

    def _update(self, session):
        for o in self._overriders:
            o.update(session)

    def reset(self, session):
        for o in self._overriders:
            o.reset(session)

    def _info(self, session):
        return self._info_tuple(overriders=self._overriders)

    def __repr__(self):
        return repr(self._overriders)
