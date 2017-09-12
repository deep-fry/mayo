import itertools
from contextlib import contextmanager
from collections import OrderedDict

import tensorflow as tf
from tensorflow.contrib import slim

from mayo.log import log
from mayo.util import (
    import_from_string, object_from_params, multi_objects_from_params, tabular)
from mayo.override import ChainOverrider


class _InstantiationParamTransformer(object):
    def __init__(self, num_classes, is_training):
        super().__init__()
        self.num_classes = num_classes
        self.is_training = is_training
        self.overriders = []

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
                # FIXME '_inherit' is pickle-initializier specific
                p = dict(p)
                for k in p.get('_inherit', []):
                    p[k] = params[k]
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

    def _config_layer(self, params):
        # activation
        fn = params.get('activation_fn', None)
        if fn is not None:
            fn = import_from_string(fn)
            params['activation_fn'] = fn
        activation_overrider = params.pop('activation_overrider', None)
        if activation_overrider:
            params['activation_fn'] = lambda x: (fn or tf.nn.relu)(
                activation_overrider.apply(tf.get_variable, x))

        # num outputs
        if params.get('num_outputs', None) == 'num_classes':
            params['num_outputs'] = self.num_classes
        # set up parameters
        params['scope'] = params.pop('name')
        try:
            params['padding'] = params['padding'].upper()
        except KeyError:
            pass

    def _norm_scope(self, params):
        # we do not have direct access to normalizer instantiation,
        # so arg_scope must be used
        norm_params = params.pop('normalizer_fn', None)
        if not norm_params:
            return slim.arg_scope([])
        obj, norm_params = object_from_params(norm_params)
        norm_params['is_training'] = self.is_training
        params['normalizer_fn'] = obj
        return slim.arg_scope([params['normalizer_fn']], **norm_params)

    def _overrider_scope(self, params):
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

        # we do not have direct access to slim.model_variable creation,
        # so arg_scope must be used
        scope = tf.get_variable_scope()
        return tf.variable_scope(scope, custom_getter=custom_getter)

    def transform(self, params):
        params = dict(params)
        # weight and bias hyperparams
        self._create_hyperobjects(params)
        # layer configs
        self._config_layer(params)
        # normalization arg_scope
        norm_scope = self._norm_scope(params)
        # overrider arg_scope
        overrider_scope = self._overrider_scope(params)
        return params, norm_scope, overrider_scope


class BaseNet(object):
    def __init__(
            self, config, images, labels, is_training,
            graph=None, reuse=None):
        super().__init__()
        self.graph = graph or tf.Graph()
        self.config = config
        self.is_training = is_training
        self._reuse = reuse
        self.end_points = OrderedDict()
        self.end_points['images'] = images
        self.end_points['labels'] = labels
        self.instantiate()

    @contextmanager
    def context(self):
        graph_ctx = self.graph.as_default()
        var_ctx = tf.variable_scope(self.config['name'], reuse=self._reuse)
        cpu_ctx = slim.arg_scope([slim.model_variable], device='/cpu:0')
        with graph_ctx, var_ctx, cpu_ctx as scope:
            yield scope

    def _add_end_point(self, key, layer):
        if key in self.end_points:
            raise KeyError(
                'layer {!r} already exists in end_points.'.format(layer))
        self.end_points[key] = layer

    def _instantiate(self):
        transformer = _InstantiationParamTransformer(
            self.config.num_classes(), self.is_training)
        transform = transformer.transform
        net = self.end_points['images']
        for params in self.config.net:
            name = params['name']
            params, norm_scope, overrider_scope = transform(params)
            # get method by its name to instantiate a layer
            func, params = object_from_params(params, self, 'instantiate_')
            # instantiation
            log.debug('Instantiating {!r} with params {}'.format(name, params))
            with norm_scope, overrider_scope:
                net = func(net, params)
            # save end points
            self._add_end_point(name, net)
            if name != 'logits' and name == self.config.logits:
                self._add_end_point('logits', net)
        # overriders
        self.overriders = transformer.overriders

    def instantiate(self):
        # force all Variables to reside on the CPU
        with self.context():
            self._instantiate()

    def generic_instantiate(self, net, params):
        raise NotImplementedError

    def update_overriders(self):
        ops = []
        for o in self.overriders:
            try:
                ops.append(o.update())
            except NotImplementedError:
                pass
        return ops

    def logits(self):
        return self.end_points['logits']

    def loss(self):
        try:
            return self.end_points['loss']
        except KeyError:
            pass
        labels = self.end_points['labels']
        logits = self.end_points['logits']
        with tf.name_scope('loss'):
            labels = slim.one_hot_encoding(labels, logits.shape[1])
            loss = tf.losses.softmax_cross_entropy(
                logits=logits, onehot_labels=labels)
            loss = tf.reduce_mean(loss)
            tf.add_to_collection('losses', loss)
        self._add_end_point('loss', loss)
        return loss

    def accuracy(self):
        try:
            return self.end_points['accuracy']
        except KeyError:
            pass
        logits = self.end_points['logits']
        labels = self.end_points['labels']
        acc = tf.nn.in_top_k(logits, labels, 1)
        self._add_end_point('accuracy', acc)
        return acc

    def save_graph(self):
        writer = tf.summary.FileWriter(self.config['name'], self.graph)
        writer.close()

    def info(self):
        def format_shape(shape):
            return ' x '.join(str(s or '?') for s in shape.as_list())

        param_table = [('Param', 'Shape', 'Count'), '-']
        total = 0
        for v in tf.trainable_variables():
            shape = v.get_shape()
            v_total = 1
            for dim in shape:
                v_total *= dim.value
            total += v_total
            param_table.append((v.name, format_shape(shape), v_total))
        param_table += ['-', (None, '    Total:', total)]
        param_table = tabular(param_table)

        layer_table = [('Layer', 'Shape'), '-']
        for name, layer in self.end_points.items():
            layer_table.append((name, format_shape(layer.shape)))
        layer_table = tabular(layer_table)
        return param_table + '\n' + layer_table


class Net(BaseNet):
    def instantiate_convolution(self, net, params):
        return slim.conv2d(net, **params)

    def instantiate_depthwise_separable_convolution(self, net, params):
        scope = params.pop('scope')
        num_outputs = params.pop('num_outputs')
        stride = params.pop('stride')
        kernel = params.pop('kernel_size')
        depth_multiplier = params.pop('depth_multiplier', 1)
        depthwise_regularizer = params.pop('depthwise_regularizer')
        pointwise_regularizer = params.pop('pointwise_regularizer')
        # depthwise layer
        depthwise = slim.separable_conv2d(
            net, num_outputs=None, kernel_size=kernel, stride=stride,
            weights_regularizer=depthwise_regularizer,
            depth_multiplier=1, scope='{}_depthwise'.format(scope), **params)
        # pointwise layer
        num_outputs = max(int(num_outputs * depth_multiplier), 8)
        pointwise = slim.conv2d(
            depthwise, num_outputs=num_outputs, kernel_size=[1, 1], stride=1,
            weights_regularizer=pointwise_regularizer,
            scope='{}_pointwise'.format(scope), **params)
        return pointwise

    @staticmethod
    def _reduce_kernel_size_for_small_input(params, tensor):
        shape = tensor.get_shape().as_list()
        if shape[1] is None or shape[2] is None:
            return
        kernel = params['kernel_size']
        if isinstance(kernel, int):
            kernel = [kernel, kernel]
        stride = params.get('stride', 1)
        params['kernel_size'] = [
            min(shape[1], kernel[0]), min(shape[2], kernel[1])]
        # tensorflow complains when stride > kernel size
        params['stride'] = min(stride, kernel[0], kernel[1])

    def instantiate_average_pool(self, net, params):
        self._reduce_kernel_size_for_small_input(params, net)
        return slim.avg_pool2d(net, **params)

    def instantiate_max_pool(self, net, params):
        self._reduce_kernel_size_for_small_input(params, net)
        return slim.max_pool2d(net, **params)

    def instantiate_fully_connected(self, net, params):
        return slim.fully_connected(net, **params)

    def instantiate_softmax(self, net, params):
        return slim.softmax(net, **params)

    def instantiate_dropout(self, net, params):
        params['is_training'] = self.is_training
        return slim.dropout(net, **params)

    def instantiate_squeeze(self, net, params):
        params['name'] = params.pop('scope')
        return tf.squeeze(net, **params)

    def instantiate_flatten(self, net, params):
        return slim.flatten(net, **params)

    def instantiate_local_response_normalization(self, net, params):
        params['name'] = params.pop('scope')
        return tf.nn.local_response_normalization(net, **params)
