import re
import pprint
import weakref
import itertools
import collections

import networkx as nx

from mayo.util import ensure_list, recursive_apply
from mayo.parse import ArithTag


def _replace_module_kwargs(params):
    kwargs = params.get('kwargs', {})
    replace_map = {
        key: params.get(key, default_value)
        for key, default_value in kwargs.items()}

    def replace_str(value):
        regex = r'\^\(([_a-zA-Z][_a-zA-Z0-9\.]*)\)'
        while True:
            keys = re.findall(regex, value, re.MULTILINE)
            if not keys:
                break
            for k in keys:
                placeholder = '^({})'.format(k)
                replace_value = replace_map[k]
                if value == placeholder:
                    return replace_value
                value = value.replace(placeholder, str(replace_value))
        return value

    def replace_arith(value):
        return ArithTag(replace_str(value.content))

    def skip_inner_module(value):
        if not isinstance(value, collections.Mapping):
            return None
        if value.get('type') != 'module':
            return None
        return value

    def replace(params, key):
        p = params[key]
        if isinstance(p, collections.Sequence):
            p = [e.asdict(eval=False) for e in p]
        else:
            p = p.asdict(eval=False)
        func_map = {str: replace_str, ArithTag: replace_arith}
        params[key] = recursive_apply(p, func_map, skip_inner_module)

    replace(params, 'layers')
    replace(params, 'graph')
    return params


class NodeBase(object):
    def __init__(self, name, module, graph):
        super().__init__()
        self.name = name
        self.module = tuple(module)
        self.graph = weakref.ref(graph)

    def __getstate__(self):
        return {
            'name': self.name,
            'module': self.module,
            'graph': None,
        }

    @property
    def predecessors(self):
        return list(self.graph().nx_graph.predecessors(self))

    @property
    def successors(self):
        return list(self.graph().nx_graph.successors(self))

    def formatted_name(self):
        return '{}/{}'.format('/'.join(self.module), self.name)

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
    def __init__(self, name, params, module, graph):
        super().__init__(name, module, graph)
        self.params = params


class MultiNodeBase(NodeBase):
    def __init__(self, nodes, module, graph):
        name = '{{{}}}'.format(', '.join(nodes))
        super().__init__(name, module, graph)


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
    """ Converts model description into a graph.  """
    def __init__(self, model):
        super().__init__()
        self.nx_graph = nx.OrderedMultiDiGraph()
        self._input_names = inputs = model.get('inputs', 'input')
        self._output_names = outputs = model.get('outputs', 'output')
        self._add_module(inputs, outputs, model['name'], model, [])
        self._optimize()
        self._validate()

    def add_edge(self, from_node, to_node):
        self.nx_graph.add_edge(from_node, to_node)
        if from_node == to_node:
            raise ValueError('Self-loop is not allowed.')
        if not isinstance(to_node, JoinNode) and len(to_node.predecessors) > 1:
            raise EdgeError(
                'Node {!r} is not a JoinNode but has multiple inputs.'
                .format(to_node))

    def input_nodes(self):
        return self._filter_nodes(
            lambda n: n.name in self._input_names and not n.module)

    def output_nodes(self):
        return self._filter_nodes(
            lambda n: n.name in self._output_names and not n.module)

    def nodes(self):
        return self.nx_graph.nodes()

    def edges(self):
        return self.nx_graph.edges()

    def has_path(self, from_node, to_node):
        return nx.has_path(self.nx_graph, from_node, to_node)

    def remove_node(self, node):
        return self.nx_graph.remove_node(node)

    def _filter_nodes(self, func):
        return [n for n in self.nodes() if func(n)]

    def tensor_nodes(self):
        return self._filter_nodes(lambda n: isinstance(n, TensorNode))

    def layer_nodes(self):
        return self._filter_nodes(lambda n: isinstance(n, LayerNode))

    def topological_order(self):
        return nx.topological_sort(self.nx_graph)

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
            with_layers = ensure_list(connection['with'] or [])
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
                        'for layer {!r}.'
                        .format(input_names, output_names, layer_name))
                layer_params = None
                if layer_name is not None:
                    try:
                        layer_params = params['layers'][layer_name]
                    except KeyError:
                        raise KeyError(
                            'Layer named {!r} is not defined.'
                            .format(layer_name))
                self._add_layer(
                    input_names, output_names,
                    layer_name, layer_params, submodule_path)
        # add interface IO
        from_nodes = []
        input_names = params.get('inputs', ['input'])
        for from_name, input_name in zip(from_names, input_names):
            from_node = TensorNode(from_name, module_path, self)
            from_nodes.append(from_node)
            input_node = TensorNode(input_name, submodule_path, self)
            self.add_edge(from_node, input_node)
        to_nodes = []
        output_names = params.get('outputs', ['output'])
        for output_name, to_name in zip(output_names, to_names):
            output_node = TensorNode(output_name, submodule_path, self)
            to_node = TensorNode(to_name, module_path, self)
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
        from_nodes = [TensorNode(n, module_path, self) for n in from_names]
        if len(from_nodes) == 1:
            join_node = from_nodes[0]
        else:
            # join input nodes
            join_node = JoinNode(from_names, module_path, self)
            for each_node in from_nodes:
                self.add_edge(each_node, join_node)
        # layer
        if layer_name is None:
            layer_node = join_node
        else:
            layer_node = LayerNode(layer_name, layer_params, module_path, self)
            self.add_edge(join_node, layer_node)
        # outputs
        to_nodes = [TensorNode(n, module_path, self) for n in to_names]
        if len(to_nodes) == 1:
            self.add_edge(layer_node, to_nodes[0])
        else:
            split_node = SplitNode(to_names, module_path, self)
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
        # remove redundant tensor nodes from graph
        for node in list(self.nodes()):
            if not isinstance(node, TensorNode):
                continue
            preds = node.predecessors
            succs = node.successors
            if not (len(preds) == len(succs) == 1):
                continue
            changed = True
            # remove current node as it is redundant
            self.remove_node(node)
            self.add_edge(preds[0], succs[0])
        # remove nodes not connected to output
        output_nodes = self.output_nodes()
        for node in list(self.nodes()):
            if not any(self.has_path(node, o) for o in output_nodes):
                self.remove_node(node)
        return changed

    def _ensure_connection(self, from_nodes, to_nodes):
        iterator = itertools.product(
            ensure_list(from_nodes), ensure_list(to_nodes))
        for i, o in iterator:
            if not any(nx.all_simple_paths(self.nx_graph, i, o)):
                undirected = self.nx_graph.to_undirected()
                subgraphs = pprint.pformat(list(
                    nx.connected_components(undirected)))
                raise GraphIOError(
                    'We expect the net to have a path from the inputs '
                    'to the outputs, a path does not exist between {} and {}. '
                    'Disjoint subgraph nodes:\n{}'.format(i, o, subgraphs))

    def _validate(self):
        # graph is acyclic
        cycles = nx.simple_cycles(self.nx_graph)
        try:
            cycle = next(cycles)
        except StopIteration:
            pass
        else:
            raise GraphCyclicError(
                'Graph is not acyclic, contains a cycle {}'.format(cycle))
