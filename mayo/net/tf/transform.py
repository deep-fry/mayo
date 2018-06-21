import copy
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
    def __init__(self, session, reuse):
        super().__init__()
        self.session = session
        self.is_training = session.is_training
        self.reuse = reuse
        self.overriders = {}
        self.variables = {}

    def _add_overrider(self, node, name, overrider):
        self.overriders.setdefault(node, {})[name] = overrider
        variables = self.variables.setdefault(node, {})
        for var in overrider.parameter_variables:
            variables[var.op.name] = var

    def _create_hyperobjects(self, layer_node, params):
        suffixes = ['regularizer', 'initializer']
        for key, p in params.items():
            if not any(key.endswith(s) for s in suffixes):
                continue
            # regularizer and initializer
            cls, p = object_from_params(p)
            params[key] = cls(**p)

        def create_overrider(overriders):
            # TODO _add_overrider here
            for name, p in overriders.items():
                if p.get('type'):
                    continue
                raise TypeError(
                    'We expect a mapping of name-overrider pairs, overrider '
                    'named {!r} does not have a type.'.format(name))
            if all(not p.get('_priority') for p in overriders.values()):
                log.warn(
                    'Priority not specified for a sequence of overriders '
                    'in layer {!r}, which may result in unexpected ordering.'
                    .format(layer_node.formatted_name()))
            overriders = list(reversed(sorted(
                overriders.values(), key=lambda p: p.get('_priority', 0))))
            overriders = [
                cls(session=self.session, **p)
                for cls, p in multi_objects_from_params(overriders)]
            if len(overriders) == 1:
                return overriders[0]
            return ChainOverrider(session=self.session, overriders=overriders)

        overrider_params = params.get('overrider', {})
        if not overrider_params:
            return
        for key, p in list(overrider_params.items()):
            if not p:
                del overrider_params[key]
                continue
            if key == 'gradient':
                for grad_key, grad_p in p.items():
                    p[grad_key] = create_overrider(grad_p)
                continue
            overrider_params[key] = create_overrider(p)

    @staticmethod
    def custom_gradient(node, name, overrider):
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
        activation_overrider = params.get('overrider.activation', None)
        if activation_overrider:
            override_fn = lambda x: activation_overrider.apply(
                node, 'activations', tf.get_variable, x)
            self._add_overrider(node, 'activation', activation_overrider)
        else:
            override_fn = None
        # produce a default ReLU activation for overrider specifically
        default_fn = tf.nn.relu if override_fn else None
        # gradient of error
        gradient_overrider = params.get('overrider.gradient.error')
        if gradient_overrider:
            self._add_overrider(node, 'gradient.error', gradient_overrider)
            gradient_name = '{}/activations/gradient'.format(
                node.formatted_name())
            gradient_func = self.custom_gradient(
                node, gradient_name, gradient_overrider)
            tf.RegisterGradient(gradient_name)(gradient_func)
            gradient_map = {'Identity': gradient_name}

            def gradient_fn(v):
                with self.session.tf_graph.gradient_override_map(gradient_map):
                    return tf.identity(v)
        else:
            gradient_fn = None
        activation_fn = params.get('activation_fn', default_fn)
        activation_params = params.pop('activation_params', {})
        if override_fn or activation_fn or gradient_fn:
            identity_fn = lambda x: x
            gradient_fn = gradient_fn or identity_fn
            override_fn = override_fn or identity_fn
            activation_fn = activation_fn or identity_fn
            params['activation_fn'] = lambda x: \
                activation_fn(override_fn(gradient_fn(x)), **activation_params)
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

        forward_overriders = params.pop('overrider', None) or {}
        gradient_overriders = forward_overriders.pop('gradient', {})

        def custom_getter(getter, name, *args, **kwargs):
            v = getter(name, *args, **kwargs)
            log.debug('Variable {} created.'.format(v))
            key = name.replace('{}/'.format(node.formatted_name()), '')
            overrider = forward_overriders.get(key)
            if overrider:
                log.debug(
                    'Overriding {!r} with {!r}.'.format(name, overrider))
                v = overrider.apply(node, name, getter, v)
                self._add_overrider(node, name, overrider)
            # gradient overrider
            overrider = gradient_overriders.get(key)
            if overrider and self.is_training:
                gradient_name = '{}/gradient'.format(name)
                gradient_func = self.custom_gradient(node, name, overrider)
                tf.RegisterGradient(gradient_name)(gradient_func)
                gradient_map = {'Identity': gradient_name}
                with self.session.tf_graph.gradient_override_map(gradient_map):
                    v = tf.identity(v)
                self._add_overrider(
                    node, 'gradient.{}'.format(name), overrider)
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
        self._create_hyperobjects(layer_node, params)
        # layer configs
        self._config_layer(layer_node, params)
        # nested scopes
        scope = self._scope(layer_node, params)
        return params, scope
