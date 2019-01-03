import functools
import contextlib

import tensorflow as tf
from tensorflow.contrib import slim

from mayo.log import log
from mayo.util import (
    import_from_string, object_from_params, multi_objects_from_params,
    compose_functions)
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
        self._overriders = {}
        self.variables = {}

    @property
    def overriders(self):
        # only return applied overriders
        overriders = {}
        for n, os in self._overriders.items():
            nos = overriders.setdefault(n, {})
            for k, o in os.items():
                if k == 'gradient':
                    nos[k] = {gk: go for gk, go in o.items() if go._applied}
                elif o._applied:
                    nos[k] = o
        return overriders

    def _create_hyperobjects(self, layer_node, params):
        suffixes = ['regularizer', 'initializer']
        for key, p in params.items():
            if not any(key.endswith(s) for s in suffixes):
                continue
            # regularizer and initializer
            if p is None:
                params[key] = None
                continue
            cls, p = object_from_params(p)
            params[key] = cls(**p)

        def create_overrider(overriders):
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
            overriders = self._overriders.setdefault(layer_node, {})
            if key == 'gradient':
                for grad_key, grad_p in p.items():
                    q = overriders.setdefault('gradient', {})
                    p[grad_key] = q[grad_key] = create_overrider(grad_p)
                continue
            overrider_params[key] = overriders[key] = create_overrider(p)

    def _apply_gradient_overrider(self, node, name, overrider, tensor):
        @tf.custom_gradient
        def wrapped(v):
            def gradient(grad):
                log.debug(
                    'Overriding the gradient of {!r} from {!r} with {!r}.'
                    .format(name, grad, overrider))
                # when overriding gradient, we are not inside any variable
                # scope, so we use the full name for overrider hyperparameter
                # instantiation
                with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                    return overrider.apply(
                        node, 'gradient', tf.get_variable, grad)
            return v, gradient
        return wrapped(tensor)

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
        activation_functions = []
        # gradient of error
        gradient_overrider = params.get('overrider.gradient.error')
        if gradient_overrider and self.is_training:
            def gradient_fn(v):
                name = '{}/errors'.format(node.formatted_name())
                return self._apply_gradient_overrider(
                    node, name, gradient_overrider, v)
            activation_functions.append(gradient_fn)
        # activation
        activation_overrider = params.get('overrider.activation', None)
        if activation_overrider:
            override_fn = lambda x: activation_overrider.apply(
                node, 'activations', tf.get_variable, x)
            activation_functions.append(override_fn)
        # produce a default ReLU activation when overriders are used
        relu_types = [
            'convolution', 'depthwise_separable_convolution',
            'fully_connected']
        default_fn = None
        if activation_functions and node.params.type in relu_types:
            default_fn = tf.nn.relu
        activation_fn = params.get('activation_fn', default_fn)
        if activation_fn:
            activation_params = params.pop('activation_params', {})
            activation_fn = functools.partial(
                activation_fn, **activation_params)
            activation_functions.append(activation_fn)
        if activation_functions:
            params['activation_fn'] = compose_functions(activation_functions)
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
        for key, overrider in gradient_overriders.items():
            if params.pop('{}_regularizer'.format(key), None):
                log.warn(
                    'Regularizer for \'{}/{}\' is for now disabled as we '
                    'override its gradient with {!r}.'
                    .format(node.formatted_name(), key, overrider))

        def custom_getter(getter, name, *args, **kwargs):
            v = getter(name, *args, **kwargs)
            log.debug('Variable {} created.'.format(v))
            key = name.replace('{}/'.format(node.formatted_name()), '')
            overrider = forward_overriders.get(key)
            if overrider:
                log.debug(
                    'Overriding {!r} with {!r}.'.format(name, overrider))
                v = overrider.apply(node, name, getter, v)
            # gradient overrider
            overrider = gradient_overriders.get(key)
            if overrider and self.is_training:
                v = self._apply_gradient_overrider(node, name, overrider, v)
            self.variables.setdefault(node, {})[key] = v
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
        params = params.asdict()
        # weight and bias hyperparams
        self._create_hyperobjects(layer_node, params)
        # layer configs
        self._config_layer(layer_node, params)
        # nested scopes
        scope = self._scope(layer_node, params)
        return params, scope
