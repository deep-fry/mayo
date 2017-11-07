from mayo.util import memoize_method
from mayo.net.graph import Graph, LayerNode, SplitNode, JoinNode


class TensorFlowNet(object):
    def __init__(self, config, images, labels, is_training, reuse=None):
        super().__init__()
        self.config = config
        self.tensors = {}
        self.graph = self._init_graph(images, labels)
        self._instantiate_from_graph(self.graph)

    def _init_graph(self, images, labels):
        graph = Graph(self.config.model)
        # check IO
        inputs = list(graph.input_nodes())
        if len(inputs) != 1:
            raise ValueError(
                'We expect the graph to have a unique input, found {} inputs.'
                .format(len(inputs)))
        images = inputs.pop()
        if images.name != 'images':
            raise ValueError(
                'We expect the input node to be named "images", found {!r}.'
                .format(images.name))
        self.tensors[images] = images
        return graph

    @memoize_method
    def _logits_node(self):
        nodes = list(self.graph.output_nodes())
        if len(nodes) != 1 or nodes[0].name != 'logits':
            raise ValueError(
                'We expect the graph to have a unique output named '
                '"logits", found {!r}.'.format(nodes))
        return nodes.pop()

    def logits(self):
        return self.tensors[self._logits_node()]

    def _instantiate_from_graph(self, graph):
        for each_node in self.topological_order():
            self._instantiate_node(graph, each_node)

    def _instantiate_node(self, graph, node):
        pred_nodes = tuple(graph.predecessors(node))
        if node in self.tensors:
            raise ValueError(
                'Node {!r} has already been assigned a value.'.format(node))
        for pred_node in pred_nodes:
            if pred_node not in self.tensors:
                raise ValueError(
                    'Node {!r} is not assigned.'.format(pred_node))
        if isinstance(node, JoinNode):
            self.tensors[node] = [self.tensors[p] for p in pred_nodes]
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
            for index, sibling in enumerate(graph.successors(pred_node)):
                if sibling != node:
                    continue
                values.append(self.tensors[pred_node][index])
            if len(values) > 1:
                raise ValueError(
                    'We do not support multiple edges from the same '
                    'node to the same node for now.')
            self.tensors[node] = values[0]
        else:
            self.tensors[node] = self.tensor[pred_node]
        # transform layer nodes with layer computations
        if isinstance(node, LayerNode):
            # instantiate layer
            self.tensors[node] = self._instantiate_layer(
                node.name, self.tensors[node], node.params)

    def _instantiate_layer(self, name, tensors, params):
        return tensors
