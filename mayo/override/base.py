from collections import Sequence, namedtuple

import tensorflow as tf

from mayo.log import log


class OverrideNotAppliedError(Exception):
    """Invoke apply before update.  """


class GetterInvokedOutsideApplyError(Exception):
    """Function getter() is invoked not in apply()."""


def _getter_not_initialized(*args, **kwargs):
    raise GetterInvokedOutsideApplyError(
        'The function `getter()` should only be invoked in `.apply()`.')


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
        self.name = None
        self.getter = _getter_not_initialized
        self.internals = {}
        self.should_update = should_update

    def _parameter(self, para_name, initial, dtype, shape, trainable=False):
        """ `tf.Variable`-based overrider hyperparameter.  """
        name = '{}/{}'.format(self.name, para_name)
        init = tf.constant_initializer(initial)
        return self.getter(
            name, dtype=dtype, shape=shape,
            initializer=init, trainable=trainable)

    def _apply(self, value):
        """
        Override this method called in `.apply()` to modify the
        variable in `value`.
        """
        raise NotImplementedError(
            'Overrider method "apply" must be implemented.')

    def apply(self, getter, value):
        """
        Things to apply to the variable in `value`, returns the
        overridden result.
        """
        def tracking_getter(name, *args, **kwargs):
            var = getter(name, *args, **kwargs)
            self.internals[name] = var
            return var
        self._applied = True
        self.name = value.op.name
        self.getter = tracking_getter
        self.before = value
        self.after = self._apply(value)
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
        if not getattr(self, '_applied', False):
            raise OverrideNotAppliedError(
                'Method "apply" must be invoked before call "update".')
        self._update(session)
        log.debug('Updated overrider {!r}'.format(self.info(session)))

    def assign(self, session):
        """Assign overridden values to parameters before overriding.  """
        session.run(tf.assign(self.before, self.after))

    def reset(self, session):
        """Reset internal variables to their respective initial values.  """
        for var in self.internals.values():
            session.run(tf.assign(var, var.initial_value))

    def _info_tuple(self, **kwargs):
        # relies on dict ordering
        cls = self.__class__.__name__
        cls_name = '{}Info'.format(cls)
        Tuple = namedtuple(cls_name, [cls] + list(kwargs))
        kwargs[cls] = self.name
        return Tuple(**kwargs)

    def info(self, session):
        return self._info_tuple()

    @classmethod
    def finalize_info(cls, table):
        pass

    def __repr__(self):
        if not self.name:
            return super().__repr__()
        return '<{} overrides {!r}>'.format(
            self.__class__.__qualname__, self.name)


class ChainOverrider(OverriderBase, Sequence):
    """ Composition of overriders.  """
    def __init__(self, overriders, should_update=True):
        super().__init__(should_update)
        self._overriders = overriders

    def __getitem__(self, index):
        return self._overriders[index]

    def __len__(self):
        return len(self._overriders)

    def _apply(self, value):
        for o in self._overriders:
            value = o.apply(self.getter, value)
        return value

    def _update(self, session):
        for o in self._overriders:
            o.update(session)

    def reset(self, session):
        for o in self._overriders:
            o.reset(session)

    def info(self, session):
        return self._info_tuple(overriders=self._overriders)

    def __repr__(self):
        return repr(self._overriders)
