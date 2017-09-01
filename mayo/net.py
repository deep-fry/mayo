from contextlib import contextmanager
from collections import OrderedDict

import tensorflow as tf
from tensorflow.contrib import slim

from mayo.log import log
from mayo.util import object_from_params, tabular


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
        getter = self._custom_getter()
        var_ctx = tf.variable_scope(
            self.config['name'], reuse=self._reuse, custom_getter=getter)
        cpu_ctx = slim.arg_scope([slim.model_variable], device='/cpu:0')
        with graph_ctx, var_ctx, cpu_ctx as scope:
            yield scope

    def _custom_getter(self, getter, *args, **kwargs):
        v = getter(*args, **kwargs)
        return self._variable_override(v)

    def _variable_override(self, variable):
        raise NotImplementedError

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
            for k in p.get('_inherit', []):
                p[k] = params[k]
            cls, p = object_from_params(p)
            params[key] = cls(**p)

        params = dict(params)
        # batch norm
        norm_params = params.pop('normalizer_fn', None)
        if norm_params:
            obj, norm_params = object_from_params(norm_params)
            norm_params['is_training'] = self.is_training
            params['normalizer_fn'] = obj
        # weight and bias hyperparams
        param_names = [
            'weights_regularizer', 'biases_regularizer',
            'weights_initializer', 'biases_initializer',
            'pointwise_regularizer', 'depthwise_regularizer']
        for name in param_names:
            create(params, name)
        # layer configs
        # num outputs
        if params.get('num_outputs', None) == 'num_classes':
            params['num_outputs'] = self.config.num_classes()
        # set up parameters
        params['scope'] = params.pop('name')
        try:
            params['padding'] = params['padding'].upper()
        except KeyError:
            pass
        return params, norm_params

    def _instantiate(self):
        net = self.end_points['images']
        for params in self.config.net:
            name = params['name']
            params, norm_params = self._instantiation_params(params)
            # we do not have direct access to normalizer instantiation,
            # so arg_scope must be used
            if norm_params:
                norm_scope = slim.arg_scope(
                    [params['normalizer_fn']], **norm_params)
            else:
                norm_scope = slim.arg_scope([])
            # get method by its name to instantiate a layer
            func, params = object_from_params(params, self, 'instantiate_')
            # instantiation
            log.debug('Instantiating {!r} with params {}'.format(name, params))
            with norm_scope:
                net = func(net, params)
            # save end points
            self._add_end_point(name, net)
            if name != 'logits' and name == self.config.logits:
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

    def instantiate_convolution_split(self, net, params):
        inputs_shape = int(net.get_shape()[-1])
        groups = params['groups']
        #  if groups <= 1:
        #      raise ValueError('Number of groups must be greater than 1.')
        if inputs_shape % groups:
            raise ValueError(
                'Shape of convolution input should be divisible by the '
                'number of groups.')
        weights_shape = params['kernel_size'] + [
            inputs_shape / params['groups'], params['num_outputs']]
        with tf.variable_scope(params['scope']), tf.device('/cpu:0'):
            weights = tf.get_variable(
                'weights', shape=weights_shape,
                initializer=params['weights_initializer'])
            biases = tf.get_variable(
                'biases', shape=[params['num_outputs']],
                initializer=params['biases_initializer'])
            strides = [1, params['stride'], params['stride'], 1]
            convolve = lambda i, k: tf.nn.conv2d(
                i, k, strides=strides, padding=params['padding'])
            # no grouping
            if groups == 1:
                return tf.nn.relu(convolve(net, weights) + biases)
            # split input and weights and convolve them separately
            input_groups = tf.split(
                axis=3, num_or_size_splits=params['groups'], value=net)
            weight_groups = tf.split(
                axis=3, num_or_size_splits=params['groups'], value=weights)
            output_groups = [
                convolve(i, k) for i, k in zip(input_groups, weight_groups)]
            # concat the convolved output together again
            conv = tf.concat(axis=3, values=output_groups)
            return tf.nn.relu(conv + biases)
