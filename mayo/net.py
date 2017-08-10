from contextlib import contextmanager

import tensorflow as tf
from tensorflow.contrib import slim

from mayo.util import import_from_dot_path


class BaseNet(object):
    def __init__(
            self, config, inputs=None, labels=None,
            batch_size=None, graph=None, reuse=None):
        super().__init__()
        self.graph = graph or tf.Graph()
        self.reuse = reuse
        self.sess = tf.Session(graph=self.graph)
        self.config = config
        self.batch_size = batch_size or config.dataset.batch_size
        self.end_points = {'inputs': inputs, 'labels': labels}
        self.instantiate()

    @contextmanager
    def context(self):
        graph_ctx = self.graph.as_default()
        var_ctx = tf.variable_scope(self.config['name'], reuse=self.reuse)
        with graph_ctx, var_ctx as scope:
            yield scope

    def instantiate(self):
        # force all Variables to reside on the CPU
        with self.context():
            self._instantiate()

    def _instantiation_params(self, params):
        def create(params, key):
            try:
                p = params[key]
            except KeyError:
                return
            cls = import_from_dot_path(p.pop('type'))
            params[key] = cls(**p)

        # layer configs
        params = dict(params)
        layer_name = params.pop('name')
        layer_type = params.pop('type')
        # set up parameters
        params['scope'] = layer_name
        # batch norm
        norm_params = params.pop('normalizer_fn', None)
        if norm_params:
            norm_type = norm_params.pop('type')
            params['normalizer_fn'] = import_from_dot_path(norm_type)
        # weight initializer
        create(params, 'weights_initializer')
        create(params, 'weights_regularizer')
        return layer_name, layer_type, params, norm_params

    def _instantiate(self):
        net = self.end_points['inputs']
        if net is None:
            # if we don't have an input, we initialize the net with
            # a placeholder input
            shape = (self.config.dataset.batch_size,) + self.config.input_shape
            net = tf.placeholder(tf.float32, shape=shape, name='input')
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
            with norm_scope:
                net = inst_func(net, params)
            # save end points
            self.end_points[layer_name] = net
            if layer_name == self.config.logits:
                self.end_points['logits'] = net

    def generic_instantiate(self, net, params):
        raise NotImplementedError(
            'Instantiation method for layer named "{}" is not implemented.'
            .format(params['scope']))

    def loss(self):
        try:
            return self.end_points['loss']
        except KeyError:
            pass
        labels = self.end_points['labels']
        if not labels:
            raise ValueError(
                'Unable to get the loss operator without "labels".')
        with tf.name_scope('loss'):
            labels = slim.one_hot_encoding(
                labels, self.config.dataset.num_classes)
            loss = tf.losses.softmax_cross_entropy(
                logits=self.end_points['logits'], onehot_labels=labels)
            loss = tf.reduce_mean(loss)
            tf.add_to_collection('losses', loss)
        self.end_points['loss'] = loss
        return loss

    def save_graph(self):
        writer = tf.summary.FileWriter(self.config['name'], self.graph)
        writer.close()


class Net(BaseNet):
    def instantiate_convolution(self, net, params):
        return slim.conv2d(net, **params)

    def instantiate_depthwise_separable_convolution(self, net, params):
        return slim.separable_conv2d(net, **params)

    @staticmethod
    def _reduced_kernel_size_for_small_input(tensor, kernel):
        shape = tensor.get_shape().as_list()
        if shape[1] is None or shape[2] is None:
            return kernel
        return [min(shape[1], kernel[0]), min(shape[2], kernel[1])]

    def instantiate_average_pool(self, net, params):
        params['kernel_size'] = self._reduced_kernel_size_for_small_input(
            net, params['kernel_size'])
        return slim.avg_pool2d(net, **params)

    def instantiate_dropout(self, net, params):
        return slim.dropout(net, **params)

    def instantiate_squeeze(self, net, params):
        params['name'] = params.pop('scope')
        return tf.squeeze(net, **params)

    def instantiate_softmax(self, net, params):
        return slim.softmax(net, **params)
