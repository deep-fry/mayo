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

    def labels(self):
        return self._labels

    def logits(self):
        return self.outputs()['output']

    @property
    def overriders(self):
        return self._transformer.overriders

    def layers(self):
        return {n.name: self._tensors[n] for n in self._graph.layer_nodes()}

    @memoize_method
    def loss(self):
        logits = self.logits()
        with tf.name_scope('loss'):
            labels = slim.one_hot_encoding(self.labels(), logits.shape[1])
            loss = tf.losses.softmax_cross_entropy(
                logits=logits, onehot_labels=labels)
            loss = tf.reduce_mean(loss)
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
        var_info = Table(['variable', 'shape'])
        var_info.add_rows((v, v.shape) for v in tf.trainable_variables())
        var_info.add_column(
            'count', lambda row: var_info[row, 'shape'].num_elements())
        var_info.set_footer(
            ['', '    total:', sum(var_info.get_column('count'))])
        layer_info = Table(['layer', 'shape'])
        for name, tensors in self.layers().items():
            tensors = ensure_list(tensors)
            for tensor in tensors:
                layer_info.add_row((name, tensor.shape))
        return {'variables': var_info, 'layers': layer_info}

    def _params_to_text(self, params):
        arguments = []
        for k, v in params.items():
            try:
                v = '{}()'.format(v.__qualname__)
            except (KeyError, AttributeError):
                pass
            arguments.append('{}={}'.format(k, v))
        return '    ' + '\n    '.join(arguments)

    def _instantiate_numeric_padding(self, tensors, params):
        pad = params.get('padding')
        if not isinstance(pad, int):
            return tensors
        # disable pad for next layer
        params['padding'] = 'VALID'
        # 4D tensor NxHxWxC, pad H and W
        paddings = [[0, 0], [pad, pad], [pad, pad], [0, 0]]
        if isinstance(tensors, collections.Sequence):
            return [tf.pad(t, paddings) for t in tensors]
        else:
            return tf.pad(tensors, paddings)

    def _instantiate_layer(self, name, tensors, params, module_path):
        # transform parameters
        params, scope = self._transformer.transform(name, params, module_path)
        with scope:
            layer_type = params['type']
            layer_key = tf.get_variable_scope().name
            log.debug(
                'Instantiating {!r} of type {!r} with arguments:\n{}'
                .format(layer_key, layer_type, self._params_to_text(params)))
            tensors = self._instantiate_numeric_padding(tensors, params)
            return super()._instantiate_layer(
                name, tensors, params, module_path)
