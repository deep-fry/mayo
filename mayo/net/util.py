import copy
import functools
import itertools
import contextlib
import collections

import tensorflow as tf
from tensorflow.contrib import slim

from mayo.log import log
from mayo.util import (
    import_from_string, object_from_params, multi_objects_from_params)
from mayo.override import ChainOverrider


def use_name_not_scope(self, params):
    params['name'] = params.pop('scope')
    return params


def one_to_one(inst_method):
    @functools.wraps(inst_method)
    def wrapper(self, tensors, params):
        if len(tensors) != 1:
            raise ValueError(
                'We expect exactly one input for {!r}'.format(inst_method))
        return [inst_method(self, tensors[0], params)]
    return wrapper


class ParameterTransformer(object):
    def __init__(self, num_classes, is_training, reuse):
        super().__init__()
        self.num_classes = num_classes
        self.is_training = is_training
        self.reuse = reuse
        self.overriders = []

    @classmethod
    def _recursive_replace(cls, value, replace):
        if isinstance(value, str):
            if value.startswith('^'):
                return replace[value[1:]]
            return value
        if isinstance(value, list):
            return [cls._recursive_replace(v, replace) for v in value]
        if isinstance(value, collections.Mapping):
            for k, v in value.items():
                value[k] = cls._recursive_replace(v, replace)
            return value
        return value

    def _repace_module_kwargs(self, params):
        if params['type'] != 'module':
            return
        kwargs = params.get('kwargs', {})
        replace = {
            key: params.get(key, default_value)
            for key, default_value in kwargs.items()}
        layers = copy.deepcopy(params['layers'])
        params['layers'] = self._recursive_replace(layers, replace)

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

    def _config_layer(self, name, params):
        # activation
        fn = params.get('activation_fn', None)
        if fn is not None:
            fn = import_from_string(fn)
            params['activation_fn'] = fn
        activation_overrider = params.pop('activation_overrider', None)
        if activation_overrider:
            params['activation_fn'] = lambda x: (fn or tf.nn.relu)(
                activation_overrider.apply(tf.get_variable, x))
        if activation_overrider:
            self.overriders.append(activation_overrider)

        # num outputs
        if params.get('num_outputs', None) == 'num_classes':
            params['num_outputs'] = self.num_classes
        # set up parameters
        params['scope'] = name
        try:
            params['padding'] = params['padding'].upper()
        except (KeyError, AttributeError):
            pass

    def _add_norm_scope(self, params, scope_list):
        # we do not have direct access to normalizer instantiation,
        # so arg_scope must be used
        norm_params = params.pop('normalizer_fn', None)
        if not norm_params:
            return
        obj, norm_params = object_from_params(norm_params)
        norm_params['is_training'] = self.is_training
        params['normalizer_fn'] = obj
        scope = slim.arg_scope([params['normalizer_fn']], **norm_params)
        scope_list.append(scope)

    def _add_var_scope(self, name, module_path, scope_list):
        if not module_path:
            return
        path = '/'.join(module_path + (name, ))
        scope = tf.variable_scope(path, reuse=self.reuse)
        scope_list.append(scope)

    def _add_overrider_scope(self, params, scope_list):
        biases_overrider = params.pop('biases_overrider', None)
        weights_overrider = params.pop('weights_overrider', None)

        def custom_getter(getter, *args, **kwargs):
            v = getter(*args, **kwargs)
            name = v.op.name
            overrider = None
            if 'biases' in name:
                overrider = biases_overrider
            elif 'weights' in name:
                overrider = weights_overrider
            if overrider is None:
                return v
            log.debug('Overriding {!r} with {!r}'.format(v.op.name, overrider))
            ov = overrider.apply(getter, v)
            self.overriders.append(overrider)
            return ov

        @contextlib.contextmanager
        def custom_scope():
            # we do not have direct access to variable creation,
            # so scope must be used
            scope = tf.get_variable_scope()
            with tf.variable_scope(scope, custom_getter=custom_getter):
                yield

        scope_list.append(custom_scope())

    @staticmethod
    @contextlib.contextmanager
    def _scope_functional(scopes):
        with contextlib.ExitStack() as scope_stack:
            for scope in scopes:
                scope_stack.enter_context(scope)
            yield

    def _scope(self, name, params, module_path):
        # scopes
        scope_list = []
        # pin variables on cpu
        cpu_context = slim.arg_scope([slim.model_variable], device='/cpu:0')
        scope_list.append(cpu_context)
        # normalization arg_scope
        self._add_norm_scope(params, scope_list)
        # variable scope
        self._add_var_scope(name, module_path, scope_list)
        # overrider custom-getter variable scope
        self._add_overrider_scope(params, scope_list)
        # custom nested scope
        return self._scope_functional(scope_list)

    def transform(self, name, params, module_path):
        params = dict(params)
        # replace module kwargs with values
        self._repace_module_kwargs(params)
        # weight and bias hyperparams
        self._create_hyperobjects(params)
        # layer configs
        self._config_layer(name, params)
        # nested scopes
        scope = self._scope(name, params, module_path)
        return params, scope
