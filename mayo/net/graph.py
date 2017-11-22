import copy
import pprint
import itertools
import collections
import networkx as nx

from mayo.util import ensure_list, recursive_apply


def _replace_module_kwargs(params):
    kwargs = params.get('kwargs', {})
    replace_map = {
        key: params.get(key, default_value)
        for key, default_value in kwargs.items()}

    def skip(value):
        if not isinstance(value, collections.Mapping):
            return None
        if value.get('type') != 'module':
            return None
        return value

    def replace(value):
        if value.startswith('^'):
            return replace_map[value[1:]]
        return value

    layers = copy.deepcopy(params['layers'])
    params['layers'] = recursive_apply(layers, {str: replace}, skip)
    return params


class NodeBase(object):
    def __init__(self, name, module):
        super().__init__()
        self.name = name
        self.module = tuple(module)

    def _eq_key(self):
        return (self.__class__, self.name, self.module)

    def __hash__(self):
        return hash(self._eq_key())

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False
        return self._eq_key() == other._eq_key()

    def __repr__(self):
        return '<{}({}, {})>'.format(
            self.__class__.__name__, self.name, '/'.join(self.module))


class TensorNode(NodeBase):
    """ A tensor-specifying node.  """


class LayerNode(NodeBase):
    """ A layer-specifying node.  """
    def __init__(self, name, params, module):
        super().__init__(name, module)
        self.params = params


class MultiNodeBase(NodeBase):
    def __init__(self, nodes, module):
        name = ', '.join(nodes)
        super().__init__(name, module)


class JoinNode(MultiNodeBase):
    """ A node to concat input nodes.  """


class SplitNode(MultiNodeBase):
    """ A node to split input nodes.  """


class GraphError(Exception):
    """ Graph construction error.  """


class GraphIOError(GraphError):
    """ Graph IO error.  """


class GraphCyclicError(GraphError):
    """ Graph is not acyclic.  """


class EdgeError(GraphError):
    """ Edge creation error.  """


class Graph(object):
    def __init__(self, model):
        super().__init__()
        self._graph = nx.OrderedMultiDiGraph()
        inputs = model.get('inputs', 'input')
        outputs = model.get('outputs', 'output')
        self._add_module(inputs, outputs, model['name'], model, [])
        self._optimize()
        self._validate()

    def add_edge(self, from_node, to_node):
        def validate_multi_edges(node):
            if not isinstance(node, JoinNode):
                if len(tuple(self.predecessors(node))) > 1:
                    raise EdgeError(
                        'Node {!r} is not a JoinNode but has multiple inputs.')
        if from_node == to_node:
            raise ValueError('Self-loop is not allowed.')
        rv = self._graph.add_edge(from_node, to_node)
        validate_multi_edges(from_node)
        validate_multi_edges(to_node)
        return rv

    def input_nodes(self):
        return self._filter_nodes(
            lambda n: len(list(self.predecessors(n))) == 0)

    def output_nodes(self):
        return self._filter_nodes(
            lambda n: len(list(self.successors(n))) == 0)

    def nodes(self):
        return self._graph.nodes()

    def edges(self):
        return self._graph.edges()

    def predecessors(self, node):
        return self._graph.predecessors(node)

    def successors(self, node):
        return self._graph.successors(node)

    def remove_node(self, node):
        return self._graph.remove_node(node)

    def _filter_nodes(self, func):
        return (n for n in self.nodes() if func(n))

    def tensor_nodes(self):
        return self._filter_nodes(lambda n: isinstance(n, TensorNode))

    def layer_nodes(self):
        return self._filter_nodes(lambda n: isinstance(n, LayerNode))

    def topological_order(self):
        return nx.topological_sort(self._graph)

    def _add_module(
            self, from_names, to_names,
            module_name, module_params, module_path=None):
        from_names = ensure_list(from_names)
        to_names = ensure_list(to_names)
        # replace kwargs in module params
        params = _replace_module_kwargs(module_params)
        # module path
        module_path = module_path or []
        submodule_path = list(module_path)
        if module_name is not None:
            submodule_path += [module_name]
        # add graph connections
        for connection in ensure_list(params['graph']):
            with_layers = ensure_list(connection['with'])
            edges = list(zip(
                [connection['from']] + with_layers,
                with_layers + [connection['to']],
                with_layers + [None]))
            for input_names, output_names, layer_name in edges:
                if input_names == output_names:
                    if layer_name is None:
                        continue
                    raise EdgeError(
                        'Input name {!r} collides with output name {!r} '
                        'for layer {!r}.'.format(layer_name))
                layer_params = params['layers'].get(layer_name)
                self._add_layer(
                    input_names, output_names,
                    layer_name, layer_params, submodule_path)
        # add interface IO
        from_nodes = []
        input_names = params.get('inputs', ['input'])
        for from_name, input_name in zip(from_names, input_names):
            from_node = TensorNode(from_name, module_path)
            from_nodes.append(from_node)
            input_node = TensorNode(input_name, submodule_path)
            self.add_edge(from_node, input_node)
        to_nodes = []
        output_names = params.get('outputs', ['output'])
        for output_name, to_name in zip(output_names, to_names):
            output_node = TensorNode(output_name, submodule_path)
            to_node = TensorNode(to_name, module_path)
            to_nodes.append(to_node)
            self.add_edge(output_node, to_node)
        # ensure connection
        self._ensure_connection(from_nodes, to_nodes)

    def _add_layer(
            self, from_names, to_names,
            layer_name, layer_params, module_path):
        from_names = ensure_list(from_names)
        to_names = ensure_list(to_names)
        if layer_params is not None and layer_params['type'] == 'module':
            # add module
            return self._add_module(
                from_names, to_names, layer_name, layer_params, module_path)
        # inputs
        from_nodes = [TensorNode(n, module_path) for n in from_names]
        if len(from_nodes) == 1:
            join_node = from_nodes[0]
        else:
            # join input nodes
            join_node = JoinNode(from_names, module_path)
            for each_node in from_nodes:
                self.add_edge(each_node, join_node)
        # layer
        if layer_name is None:
            layer_node = join_node
        else:
            layer_node = LayerNode(layer_name, layer_params, module_path)
            self.add_edge(join_node, layer_node)
        # outputs
        to_nodes = [TensorNode(n, module_path) for n in to_names]
        if len(to_nodes) == 1:
            self.add_edge(layer_node, to_nodes[0])
        else:
            split_node = SplitNode(to_names, module_path)
            self.add_edge(layer_node, split_node)
            for each_node in to_nodes:
                self.add_edge(split_node, each_node)

    def _optimize(self):
        while True:
            changed = self._optimize_propagation()
            if not changed:
                return

    def _optimize_propagation(self):
        changed = False
        for node in list(self.nodes()):
            if not isinstance(node, TensorNode):
                continue
            preds = tuple(self.predecessors(node))
            succs = tuple(self.successors(node))
            if not (len(preds) == len(succs) == 1):
                continue
            changed = True
            # remove current node as it is redundant
            self.remove_node(node)
            self.add_edge(preds[0], succs[0])
        return changed

    def _ensure_connection(self, from_nodes, to_nodes):
        iterator = itertools.product(
            ensure_list(from_nodes), ensure_list(to_nodes))
        for i, o in iterator:
            if not any(nx.all_simple_paths(self._graph, i, o)):
                undirected = self._graph.to_undirected()
                subgraphs = pprint.pformat(list(
                    nx.connected_components(undirected)))
                raise GraphIOError(
                    'We expect the net to have a path from the inputs '
                    'to the outputs, a path does not exist between {} and {}. '
                    'Disjoint subgraph nodes:\n{}'.format(i, o, subgraphs))

    def _validate(self):
        # graph is acyclic
        cycles = nx.simple_cycles(self._graph)
        try:
            cycle = next(cycles)
        except StopIteration:
            pass
        else:
            raise GraphCyclicError(
                'Graph is not acyclic, contains a cycle {}'.format(cycle))
