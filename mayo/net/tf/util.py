import copy
import itertools
import contextlib

import tensorflow as tf
from tensorflow.contrib import slim

from mayo.log import log
from mayo.util import (
    import_from_string, object_from_params, multi_objects_from_params)
from mayo.override import ChainOverrider


def use_name_not_scope(params):
    params['name'] = params.pop('scope')
    return params


class ParameterTransformer(object):
    def __init__(self, session, num_classes, reuse):
        super().__init__()
        self.session = session
        self.num_classes = num_classes
        self.is_training = session.is_training
        self.reuse = reuse
        self.overriders = []
        self.variables = {}

    def _create_hyperobjects(self, params):
        suffixes = ['regularizer', 'initializer', 'overrider']
        for key, p in params.items():
            if not any(key.endswith(s) for s in suffixes):
                continue
            if not key.endswith('overrider'):
                # regularizer and initializer
                cls, p = object_from_params(p)
                params[key] = cls(**p)
                continue
            # overrider
            overriders = [
                cls(session=self.session, **p)
                for cls, p in multi_objects_from_params(p)]
            if len(overriders) == 1:
                params[key] = overriders[0]
            else:
                params[key] = ChainOverrider(
                    session=self.session, overriders=overriders)

    def _config_layer(self, node, params):
        # normalizer_fn and activation_fn
        for key in ['activation_fn', 'normalizer_fn']:
            if key not in params:
                continue
            fn = params[key]
            if isinstance(fn, str):
                fn = import_from_string(fn)
            params[key] = fn
        # insert is_training into normalizer_params
        if params.get('normalizer_fn', None):
            norm_params = params.setdefault('normalizer_params', {})
            norm_params['is_training'] = self.is_training
        # activation
        activation_overrider = params.pop('activation_overrider', None)
        if activation_overrider:
            activation_fn = params.get('activation_fn', tf.nn.relu)
            if activation_fn is None:
                activation_fn = lambda x: x
            params['activation_fn'] = lambda x: activation_fn(
                activation_overrider.apply(
                    node, 'activations', tf.get_variable, x))
            self.overriders.append(activation_overrider)
        # num outputs
        if params.get('num_outputs', None) == 'num_classes':
            params['num_outputs'] = self.num_classes
        # set up other parameters
        params['scope'] = node.name
        try:
            params['padding'] = params['padding'].upper()
        except (KeyError, AttributeError):
            pass

    def _add_var_scope(self, node, params, scope_list):
        path = '/'.join(node.module)
        if not path:
            raise ValueError('Module path is empty.')

        forward_overriders = params.pop('overrider', {})
        gradient_overriders = params.pop('gradient_overrider', {})

        def custom_gradient(name, overrider):
            def wrapped(op, grad):
                log.debug(
                    'Overriding the gradient of {!r} from {!r} with {!r}.'
                    .format(name, grad, overrider))
                # when overriding gradient, we are not inside any variable
                # scope, so we use the full name for overrider hyperparameter
                # instantiation
                scope = '{}/gradient'.format(name)
                return overrider.apply(node, scope, tf.get_variable, grad)
            return wrapped

        def custom_getter(getter, name, *args, **kwargs):
            v = getter(name, *args, **kwargs)
            log.debug('Variable {} created.'.format(v))
            key = name.replace('{}/'.format(node.formatted_name()), '')
            overrider = forward_overriders.get(key)
            if overrider:
                log.debug(
                    'Overriding {!r} with {!r}.'.format(name, overrider))
                v = overrider.apply(node, name, getter, v)
                self.overriders.append(overrider)
            # gradient overrider
            overrider = gradient_overriders.get(key)
            if overrider and self.is_training:
                gradient_name = '{}/gradient'.format(name)
                gradient_func = custom_gradient(name, overrider)
                tf.RegisterGradient(gradient_name)(gradient_func)
                gradient_map = {'Identity': gradient_name}
                with self.session.tf_graph.gradient_override_map(gradient_map):
                    v = tf.identity(v)
                self.overriders.append(overrider)
            self.variables.setdefault(node, {})[name] = v
            return v

        @contextlib.contextmanager
        def custom_scope():
            # we do not have direct access to variable creation,
            # so scope must be used.
            # FIXME there is currently no possible workaround for
            # auto-generated `name_scope` from `variable_scope` with names that
            # are being uniquified.  See #39.
            var_scope = tf.variable_scope(
                path, reuse=self.reuse, custom_getter=custom_getter)
            with var_scope as scope:
                yield scope

        scope_list.append(custom_scope())

    @staticmethod
    @contextlib.contextmanager
    def _scope_functional(scopes):
        with contextlib.ExitStack() as scope_stack:
            for scope in scopes:
                scope_stack.enter_context(scope)
            yield

    def _scope(self, layer_node, params):
        # scopes
        scope_list = []
        # pin variables on cpu
        cpu_context = slim.arg_scope([slim.model_variable], device='/cpu:0')
        scope_list.append(cpu_context)
        # variable scope with custom getter for overriders
        self._add_var_scope(layer_node, params, scope_list)
        # custom nested scope
        return self._scope_functional(scope_list)

    def transform(self, layer_node, params):
        params = copy.deepcopy(params)
        # weight and bias hyperparams
        self._create_hyperobjects(params)
        # layer configs
        self._config_layer(layer_node, params)
        # nested scopes
        scope = self._scope(layer_node, params)
        return params, scope
