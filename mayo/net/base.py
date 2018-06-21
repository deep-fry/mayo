import collections

from mayo.util import object_from_params
from mayo.net.graph import Graph, TensorNode, LayerNode, SplitNode, JoinNode


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

    def layers(self):
        return {n: self._tensors[n] for n in self._graph.layer_nodes()}

    @property
    def shapes(self):
        raise NotImplementedError

    def info(self):
        return {}

    def _get_analyzer(self, analyzer_map, node):
        if node in self._graph.input_nodes():
            func = analyzer_map.get('input')
        else:
            func = analyzer_map.get(type(node))
        type_map = {
            TensorNode: 'tensor',
            JoinNode: 'join',
            SplitNode: 'split',
            LayerNode: 'layer',
        }
        func = func or analyzer_map.get(type_map[type(node)])
        if not func:
            return lambda node, prev: prev
        return func

    def _node_analysis(self, node, analyzer_map, info):
        analyzer = self._get_analyzer(analyzer_map, node)
        if node in self._graph.input_nodes():
            info[node] = analyzer(node, info.get(node, {}))
            return
        pred_nodes = node.predecessors
        if node in info:
            raise ValueError(
                'Node {!r} has already been assigned a value.'.format(node))
        for pred_node in pred_nodes:
            if pred_node not in info:
                raise ValueError(
                    'Node {!r} is not assigned.'.format(pred_node))
        if isinstance(node, JoinNode):
            info[node] = analyzer(node, [info[p] for p in pred_nodes])
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
                values.append(info[pred_node][index])
            if len(values) > 1:
                raise ValueError(
                    'We do not support multiple edges from the same '
                    'node to the same node for now.')
            info[node] = analyzer(node, values[0])
            return
        info[node] = info[pred_node]
        if isinstance(node, TensorNode):
            info[node] = analyzer(node, info[node])
        elif isinstance(node, LayerNode):
            info[node] = analyzer(node, info[node])
        else:
            raise TypeError('Unexpected node type {!r}.'.format(node))

    def dataflow_analysis(self, analyzer_map, info=None):
        info = info or {}
        for node in self._graph.topological_order():
            self._node_analysis(node, analyzer_map, info)
        return info

    def _instantiate(self):
        func_map = {'layer': self._instantiate_layer}
        self._tensors = self.dataflow_analysis(func_map, self._tensors)

    def _instantiate_layer(self, node, tensors):
        func, params = object_from_params(node.params, self, 'instantiate_')
        # instantiation
        return func(node, tensors, params)

    def estimate(self):
        func_map = {'layer': self._estimate_layer}
        return self.dataflow_analysis(func_map)

    def _estimate_layer(self, node, info):
        input_shape = [self.shapes[p] for p in node.predecessors]
        input_shape = input_shape[0] if len(input_shape) == 1 else input_shape
        output_shape = self.shapes[node]
        try:
            func, params = object_from_params(node.params, self, 'estimate_')
        except NotImplementedError:
            func = self.generic_estimate
            params = node.params
        return func(node, info, input_shape, output_shape, params)

    def generic_estimate(self, node, info, input_shape, output_shape, params):
        # disallow information before any layer to pass through
        # the layer by default
        return {}
