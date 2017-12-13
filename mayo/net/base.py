import collections

from mayo.util import object_from_params
from mayo.net.graph import Graph, LayerNode, SplitNode, JoinNode


class NetBase(object):
    def __init__(self, model, inputs):
        super().__init__()
        self._tensors = collections.OrderedDict()
        self._init_graph(model, inputs)
        self._instantiate()

    def _init_graph(self, model, inputs):
        self._graph = Graph(model)
        # initialize inputs
        input_nodes = self._graph.input_nodes()
        for n in input_nodes:
            self._tensors[n] = inputs[n.name]

    def inputs(self):
        return {n.name: self._tensors[n] for n in self._graph.input_nodes()}

    def outputs(self):
        return {n.name: self._tensors[n] for n in self._graph.output_nodes()}

    def _instantiate(self):
        for each_node in self._graph.topological_order():
            self._instantiate_node(each_node)

    def _instantiate_node(self, node):
        if node in self._graph.input_nodes():
            if node not in self._tensors:
                raise ValueError(
                    'Input node {!r} is not initialized with a value '
                    'before instantiating the net.'.format(node))
            return
        pred_nodes = node.predecessors
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
            for index, sibling in enumerate(pred_node.successors):
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
                node, self._tensors[node], node.params)

    def _instantiate_layer(self, node, tensors, params):
        func, params = object_from_params(params, self, 'instantiate_')
        # instantiation
        return func(node, tensors, params)
