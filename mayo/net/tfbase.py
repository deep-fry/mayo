import collections

import tensorflow as tf
from tensorflow.contrib import slim

from mayo.log import log
from mayo.util import memoize_method, Table, ensure_list
from mayo.net.base import NetBase
from mayo.net.util import ParameterTransformer


class TFNetBase(NetBase):
    """ Utility functions to create a TensorFlow network.  """

    def __init__(self, model, images, labels, num_classes, is_training, reuse):
        self.is_training = is_training
        self._transformer = ParameterTransformer(
            num_classes, is_training, reuse)
        self.update_functions = {}
        super().__init__(model, {'input': images})
        self._labels = labels
        self._verify_io()

    def _verify_io(self):
        nodes = list(self._graph.input_nodes())
        if len(nodes) != 1 and nodes[0].name != 'input':
            raise ValueError(
                'We expect the graph to have a unique images input named '
                '"input", found {!r}.'.format(nodes))
        nodes = list(self._graph.output_nodes())
        if len(nodes) != 1 or nodes[0].name != 'output':
            raise ValueError(
                'We expect the graph to have a unique logits output named '
                '"output", found {!r}.'.format(nodes))

    def register_update(self, collection, tensor, function):
        # register function to register tensor in progress update
        tf.add_to_collection(collection, tensor)
        self.update_functions[collection] = function

    def labels(self):
        return self._labels

    def logits(self):
        return self.outputs()['output']

    @property
    def overriders(self):
        return self._transformer.overriders

    def layers(self):
        layers = {}
        for n in self._graph.layer_nodes():
            name = '{}/{}'.format('/'.join(n.module), n.name)
            layers[name] = self._tensors[n]
        return layers

    @memoize_method
    def loss(self):
        logits = self.logits()
        with tf.name_scope('loss'):
            labels = slim.one_hot_encoding(self.labels(), logits.shape[1])
            loss = tf.losses.softmax_cross_entropy(
                logits=logits, onehot_labels=labels)
        return loss

    def top(self, count=1):
        name = 'top_{}'.format(count)
        try:
            return self._tensors[name]
        except KeyError:
            pass
        top = tf.nn.in_top_k(self.logits(), self.labels(), count)
        self._tensors[name] = top
        return top

    def accuracy(self, top_n=1):
        name = 'accuracy_{}'.format(top_n)
        try:
            return self._tensors[name]
        except KeyError:
            pass
        top = self.top(top_n)
        acc = tf.reduce_sum(tf.cast(top, tf.float32))
        acc /= top.shape.num_elements()
        self._tensors[name] = acc
        return acc

    def info(self):
        # trainable table
        trainable_vars = tf.trainable_variables()
        trainable = Table(['trainable', 'shape'])
        trainable.add_rows((v, v.shape) for v in trainable_vars)
        trainable.add_column(
            'count', lambda row: trainable[row, 'shape'].num_elements())
        trainable.set_footer(
            ['', '    total:', sum(trainable.get_column('count'))])

        # nontrainable table
        nontrainable = Table(['nontrainable', 'shape'])
        for var in tf.global_variables():
            if var not in trainable_vars:
                nontrainable.add_row((var, var.shape))

        # layers
        layer_info = Table(['layer', 'shape'])
        for name, tensors in self.layers().items():
            tensors = ensure_list(tensors)
            for tensor in tensors:
                layer_info.add_row((name, tensor.shape))

        return {
            'trainables': trainable,
            'nontrainables': nontrainable,
            'layers': layer_info,
        }

    def _params_to_text(self, params):
        arguments = []
        for k, v in params.items():
            try:
                v = '{}()'.format(v.__qualname__)
            except (KeyError, AttributeError):
                pass
            arguments.append('{}={}'.format(k, v))
        return '    ' + '\n    '.join(arguments)

    def _instantiate_layer(self, name, tensors, params, module_path):
        # transform parameters
        params, scope = self._transformer.transform(name, params, module_path)
        with scope:
            layer_type = params['type']
            layer_key = '{}/{}'.format(
                tf.get_variable_scope().name, params['scope'])
            layer_args = self._params_to_text(params)
            log.debug(
                'Instantiating {!r} of type {!r} with arguments:\n{}'
                .format(layer_key, layer_type, layer_args))
            tensors = self.instantiate_numeric_padding(tensors, params)
            layer = super()._instantiate_layer(
                name, tensors, params, module_path)
        return layer

    def instantiate_numeric_padding(self, tensors, params):
        pad = params.get('padding')
        if pad is None or isinstance(pad, str):
            return tensors
        # 4D tensor NxHxWxC, pad H and W
        if isinstance(pad, int):
            paddings = [[0, 0], [pad, pad], [pad, pad], [0, 0]]
        elif isinstance(pad, collections.Sequence):
            pad_h, pad_w = pad
            paddings = [[0, 0], pad_h, pad_w, [0, 0]]
        else:
            raise ValueError(
                'We do not know what to do with a padding {!r}, we accept an '
                'integer, a string or a sequence of height and width paddings '
                '[pad_h, pad_w].'.format(pad))
        # disable pad for next layer
        params['padding'] = 'VALID'
        if isinstance(tensors, collections.Sequence):
            return [tf.pad(t, paddings) for t in tensors]
        else:
            return tf.pad(tensors, paddings)
