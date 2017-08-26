from contextlib import contextmanager
from collections import OrderedDict

import tensorflow as tf
from tensorflow.contrib import slim

from mayo.util import import_from_string, tabular


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

    def _instantiation_params(self, params):
        def create(params, key):
            p = params.get(key, None)
            if p is None:
                return
            if p is None:
                return
            p = dict(p)
            cls = import_from_string(p.pop('type'))
            for k in p.pop('_inherit', []):
                p[k] = params[k]
            params[key] = cls(**p)

        params = dict(params)
        # batch norm
        norm_params = params.pop('normalizer_fn', None)
        if norm_params:
            norm_params = dict(norm_params)
            norm_type = norm_params.pop('type')
            norm_params['is_training'] = self.is_training
            params['normalizer_fn'] = import_from_string(norm_type)
        # weight and bias hyperparams
        param_names = [
            'weights_regularizer', 'biases_regularizer',
            'weights_initializer', 'biases_initializer',
            'pointwise_regularizer', 'depthwise_regularizer']
        for name in param_names:
            create(params, name)
        # layer configs
        layer_name = params.pop('name')
        layer_type = params.pop('type')
        # set up parameters
        params['scope'] = layer_name
        try:
            params['padding'] = params['padding'].upper()
        except KeyError:
            pass
        return layer_name, layer_type, params, norm_params

    def _instantiate(self):
        net = self.end_points['images']
        for params in self.config.net:
            layer_name, layer_type, params, norm_params = \
                self._instantiation_params(params)
            # get method by its name to instantiate a layer
            func_name = 'instantiate_' + layer_type
            inst_func = getattr(self, func_name, self.generic_instantiate)
            # we do not have direct access to normalizer instantiation,
            # so arg_scope must be used
            if norm_params:
                norm_scope = slim.arg_scope(
                    [params['normalizer_fn']], **norm_params)
            else:
                norm_scope = slim.arg_scope([])
            # instantiation
            try:
                with norm_scope:
                    net = inst_func(net, params)
            except NotImplementedError:
                raise NotImplementedError(
                    'Instantiation method for layer named "{}" with type "{}" '
                    'is not implemented.'.format(params['scope'], layer_type))
            # save end points
            self._add_end_point(layer_name, net)
            if layer_name != 'logits' and layer_name == self.config.logits:
                self._add_end_point('logits', net)

    def instantiate(self):
        # force all Variables to reside on the CPU
        with self.context():
            self._instantiate()

    def generic_instantiate(self, net, params):
        raise NotImplementedError

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

    def save_graph(self):
        writer = tf.summary.FileWriter(self.config['name'], self.graph)
        writer.close()

    def info(self):
        def format_shape(shape):
            return ' x '.join(
                '?' if s is None else str(s) for s in shape.as_list())

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
