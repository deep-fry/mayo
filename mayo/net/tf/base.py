import collections

import tensorflow as tf
from tensorflow.contrib import slim

from mayo.log import log
from mayo.util import memoize_method, Table, object_from_params
from mayo.net.base import NetBase
from mayo.net.tf.util import ParameterTransformer


class TFNetBase(NetBase):
    """ Utility functions to create a TensorFlow network.  """

    def __init__(self, session, model, images, labels, num_classes, reuse):
        self.session = session
        self.is_training = session.is_training
        self._transformer = ParameterTransformer(session, num_classes, reuse)
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

    @property
    def variables(self):
        return self._transformer.variables

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
        info_dict = super().info()
        # trainable table
        trainable_vars = tf.trainable_variables()
        trainable = Table(['trainable', 'shape'])
        trainable.add_rows((v, v.shape) for v in trainable_vars)
        trainable.add_column(
            'count', lambda row: trainable[row, 'shape'].num_elements())
        trainable.set_footer(
            ['', '    total:', sum(trainable.get_column('count'))])
        info_dict['trainables'] = trainable
        # nontrainable table
        nontrainable = Table(['nontrainable', 'shape'])
        for var in tf.global_variables():
            if var not in trainable_vars:
                nontrainable.add_row((var, var.shape))
        info_dict['nontrainables'] = nontrainable
        return info_dict

    @property
    def shapes(self):
        unify = lambda t: tuple(int(s) for s in t.shape)
        shapes = {}
        for node, tensors in self._tensors.items():
            if isinstance(tensors, collections.Sequence):
                shapes[node] = [unify(t) for t in tensors]
            else:
                shapes[node] = unify(tensors)
        return shapes

    def _params_to_text(self, params):
        arguments = []
        for k, v in params.items():
            try:
                v = '{}()'.format(v.__qualname__)
            except (KeyError, AttributeError):
                pass
            arguments.append('{}={}'.format(k, v))
        return '    ' + '\n    '.join(arguments)

    def _instantiate_layer(self, node, tensors, params):
        # transform parameters
        params, scope = self._transformer.transform(node, params)
        with scope:
            tensors = self.instantiate_numeric_padding(node, tensors, params)
            layer_type = params['type']
            layer_key = '{}/{}'.format(
                tf.get_variable_scope().name, params['scope'])
            layer_args = self._params_to_text(params)
            log.debug(
                'Instantiating {!r} of type {!r} with arguments:\n{}\n'
                '  for tensor(s) {}.'
                .format(layer_key, layer_type, layer_args, tensors))
            # get method by its name to instantiate a layer
            try:
                func, params = object_from_params(params, self, 'instantiate_')
            except NotImplementedError:
                func = self.generic_instantiate
            # instantiation
            layer = func(node, tensors, params)
        return layer

    def generic_instantiate(self, node, tensors, params):
        raise NotImplementedError(
            '{!r} does not know how to instantiate layer with type {!r}.'
            .format(self, params['type']))

    def instantiate_numeric_padding(self, node, tensors, params):
        pad = params.get('padding')
        if pad is None or isinstance(pad, str):
            return tensors
        # 4D tensor NxHxWxC, pad H and W
        if isinstance(pad, int):
            paddings = [[0, 0], [pad, pad], [pad, pad], [0, 0]]
        elif isinstance(pad, collections.Sequence):
            pad_h, pad_w = pad
            paddings = [[0, 0], [pad_h] * 2, [pad_w] * 2, [0, 0]]
        else:
            raise ValueError(
                'We do not know what to do with a padding {!r}, we accept an '
                'integer, a string or a sequence of height and width paddings '
                '[pad_h, pad_w].'.format(pad))
        # disable pad for next layer
        params['padding'] = 'VALID'
        log.debug(
            'Instantiating padding {!r} for tensors(s) {}.'
            .format(paddings, tensors))
        if isinstance(tensors, collections.Sequence):
            return [tf.pad(t, paddings) for t in tensors]
        return tf.pad(tensors, paddings)
