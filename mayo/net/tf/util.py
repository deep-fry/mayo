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
    def __init__(self, num_classes, is_training, reuse):
        super().__init__()
        self.num_classes = num_classes
        self.is_training = is_training
        self.reuse = reuse
        self.overriders = []
        self.variables = {}

    def _create_hyperobjects(self, params):
        def _create_object_for_key(params, key):
            p = params.get(key, None)
            if p is None:
                return
            if 'overrider' in key:
                overriders = [
                    cls(**p) for cls, p in multi_objects_from_params(p)]
                if len(overriders) == 1:
                    params[key] = overriders[0]
                else:
                    params[key] = ChainOverrider(overriders)
            else:
                cls, p = object_from_params(p)
                params[key] = cls(**p)

        var_names = ['weights', 'biases']
        obj_names = ['regularizer', 'initializer', 'overrider']
        param_names = [
            '{}_{}'.format(v, o)
            for v, o in itertools.product(var_names, obj_names)]
        param_names += [
            'pointwise_regularizer', 'depthwise_regularizer',
            'activation_overrider']
        for name in param_names:
            _create_object_for_key(params, name)

    @staticmethod
    def _apply_overrider(component, overrider, tensor):
        # apply overrider within the given scope named by component, so as
        # to create variables that are specific to that overrider to avoid
        # collision within the same layer.
        with tf.variable_scope(component):
            return overrider

    def _config_layer(self, name, params):
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
            params['activation_fn'] = lambda x: (fn or tf.nn.relu)(
                activation_overrider.apply('activations', tf.get_variable, x))
            self.overriders.append(activation_overrider)
        # num outputs
        if params.get('num_outputs', None) == 'num_classes':
            params['num_outputs'] = self.num_classes
        # set up other parameters
        params['scope'] = name
        try:
            params['padding'] = params['padding'].upper()
        except (KeyError, AttributeError):
            pass

    def _add_var_scope(self, layer_node, params, scope_list):
        path = '/'.join(layer_node.module)
        if not path:
            raise ValueError('Module path is empty.')

        biases_overrider = params.pop('biases_overrider', None)
        weights_overrider = params.pop('weights_overrider', None)

        def custom_getter(getter, *args, **kwargs):
            v = getter(*args, **kwargs)
            name = v.op.name
            overrider = None
            if name.endswith('biases'):
                overrider = biases_overrider
            elif name.endswith('weights'):
                overrider = weights_overrider
            if overrider:
                log.debug('Overriding {!r} with {!r}'.format(name, overrider))
                v = overrider.apply(name, getter, v)
                self.overriders.append(overrider)
            node_name = layer_node.formatted_name()
            var_name = name.replace('{}/'.format(node_name), '')
            self.variables.setdefault(layer_node, {})[var_name] = v
            return v

        @contextlib.contextmanager
        def custom_scope():
            # we do not have direct access to variable creation,
            # so scope must be used
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
        params = dict(params)
        # weight and bias hyperparams
        self._create_hyperobjects(params)
        # layer configs
        self._config_layer(layer_node.name, params)
        # nested scopes
        scope = self._scope(layer_node, params)
        return params, scope
