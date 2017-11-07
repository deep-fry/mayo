import collections

import tensorflow as tf
from tensorflow.contrib import slim

from mayo.log import log
from mayo.util import memoize_method, object_from_params, Table
from mayo.net.util import ParameterTransformer
from mayo.net.graph import Graph, LayerNode, SplitNode, JoinNode


class NetBase(object):
    def __init__(self, model, images, labels, num_classes, is_training, reuse):
        super().__init__()
        self._images = images
        self._labels = labels
        self.num_classes = num_classes
        self.is_training = is_training
        self.reuse = reuse
        self._tensors = collections.OrderedDict()
        self._transformer = ParameterTransformer(
            num_classes, is_training, reuse)
        self._init_graph(model, images)
        self._instantiate()

    def _init_graph(self, model, images):
        self._graph = Graph(model)
        self._tensors[self._images_node()] = images

    def _instantiate(self):
        for each_node in self._graph.topological_order():
            self._instantiate_node(each_node)

    def _instantiate_node(self, node):
        if node in self._graph.input_nodes():
            if node not in self._tensors:
                raise ValueError(
                    'Input node {!r} is not initialized with a value '
                    'before instantiating the net.'.format(n))
            return
        pred_nodes = tuple(self._graph.predecessors(node))
        if node in self._tensors:
            raise ValueError(
                'Node {!r} has already been assigned a value.'.format(node))
        for pred_node in pred_nodes:
            if pred_node not in self._tensors:
                raise ValueError(
                    'Node {!r} is not assigned.'.format(pred_node))
        if isinstance(node, JoinNode):
            self._tensors[node] = [self._tensors[p] for p in pred_nodes]
            return
        # non-JoinNode should have only one predecessor
        # and propagate the value
        if len(pred_nodes) > 1:
            raise IndexError(
                'Number of predecessors of {!r} must be 1 '
                'for a {!r}.'.format(node.__class__.__name__))
        pred_node = pred_nodes[0]
        if isinstance(pred_node, SplitNode):
            values = []
            for index, sibling in enumerate(self._graph.successors(pred_node)):
                if sibling != node:
                    continue
                values.append(self._tensors[pred_node][index])
            if len(values) > 1:
                raise ValueError(
                    'We do not support multiple edges from the same '
                    'node to the same node for now.')
            self._tensors[node] = values[0]
        else:
            self._tensors[node] = self._tensors[pred_node]
        # transform layer nodes with layer computations
        if isinstance(node, LayerNode):
            # instantiate layer
            self._tensors[node] = self._instantiate_layer(
                node.name, self._tensors[node], node.params, node.module)

    def _instantiate_numeric_padding(self, tensors, params):
        pad = params.get('padding')
        if not isinstance(pad, int):
            return tensors
        # disable pad for next layer
        params['padding'] = 'VALID'
        # 4D tensor NxHxWxC, pad H and W
        paddings = [[0, 0], [pad, pad], [pad, pad], [0, 0]]
        return [tf.pad(t, paddings) for t in tensors]

    def _params_to_text(self, params):
        arguments = '    '
        for k, v in params.items():
            try:
                v = '{}()'.format(v.__qualname__)
            except (KeyError, AttributeError):
                pass
            arguments += '\n    {}={}'.format(k, v)
        return arguments

    def _instantiate_layer(self, name, tensors, params, module_path):
        layer_type = params['type']
        # transform parameters
        params, scope = self._transformer.transform(name, params, module_path)
        # get method by its name to instantiate a layer
        func, params = object_from_params(params, self, 'instantiate_')
        # instantiation
        with scope:
            layer_key = tf.get_variable_scope().name
            log.debug(
                'Instantiating {!r} of type {!r} with arguments:\n{}'
                .format(layer_key, layer_type, self._params_to_text(params)))
            tensors = self._instantiate_numeric_padding(tensors, params)
            return func(tensors, params)

    @property
    def overriders(self):
        return self._transformer.overriders

    @memoize_method
    def _images_node(self):
        nodes = list(self._graph.input_nodes())
        if len(nodes) != 1 and nodes[0].name != 'input':
            raise ValueError(
                'We expect the graph to have a unique images input named '
                '"input", found {!r}.'.format(nodes))
        return nodes.pop()

    @memoize_method
    def _logits_node(self):
        nodes = list(self._graph.output_nodes())
        if len(nodes) != 1 or nodes[0].name != 'output':
            raise ValueError(
                'We expect the graph to have a unique logits output named '
                '"output", found {!r}.'.format(nodes))
        return nodes.pop()

    def logits(self):
        return self._tensors[self._logits_node()]

    @memoize_method
    def loss(self):
        logits = self.logits()
        with tf.name_scope('loss'):
            labels = slim.one_hot_encoding(self._labels, logits.shape[1])
            loss = tf.losses.softmax_cross_entropy(
                logits=logits, onehot_labels=labels)
            loss = tf.reduce_mean(loss)
            tf.add_to_collection('losses', loss)
        return loss

    def top(self, count=1):
        name = 'top_{}'.format(count)
        try:
            return self._tensors[name]
        except KeyError:
            pass
        top = tf.nn.in_top_k(self.logits(), self._labels, count)
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
        for name, tensors in self.layers.items():
            for tensor in tensors:
                layer_info.add_row((name, tensor.shape))
        return {'variables': var_info, 'layers': layer_info}
