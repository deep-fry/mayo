import copy
import collections
import networkx as nx

from mayo.util import ensure_list


def _replace_module_kwargs(params):
    def recursive_replace(value, replace):
        if isinstance(value, str):
            if value.startswith('^'):
                return replace[value[1:]]
            return value
        if isinstance(value, list):
            return [recursive_replace(v, replace) for v in value]
        if isinstance(value, collections.Mapping):
            if value.get('type') == 'module':
                # skip inner modules
                return value
            for k, v in value.items():
                value[k] = recursive_replace(v, replace)
            return value
        return value

    kwargs = params.get('kwargs', {})
    replace = {
        key: params.get(key, default_value)
        for key, default_value in kwargs.items()}
    layers = copy.deepcopy(params['layers'])
    params['layers'] = recursive_replace(layers, replace)
    return params


class NodeBase(object):
    def __init__(self, name, module):
        super().__init__()
        self.name = name
        self.module = tuple(module)
        self.value = None

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


class NetBase(object):
    def __init__(self, net_name, layer_defs, graph_defs):
        super().__init__()
        self._graph = nx.OrderedMultiDiGraph()
        inputs = ['images']
        outputs = ['logits']
        params = {
            'layers': layer_defs,
            'graph': ensure_list(graph_defs),
            'inputs': inputs,
            'outputs': outputs,
        }
        self._add_module(inputs, outputs, net_name, params, [])
        self._optimize()
        self._validate()

    def add_edge(self, from_node, to_node):
        def validate_multi_edges(node):
            if not isinstance(node, JoinNode):
                if len(tuple(self.predecessors(node))) > 1:
                    raise EdgeError(
                        'Node {!r} is not a JoinNode but has multiple inputs.')
            if not isinstance(node, SplitNode):
                if len(tuple(self.successors(node))) > 1:
                    raise EdgeError(
                        'Node {!r} is not a SplitNode but has '
                        'multiple outputs.')
        if from_node == to_node:
            raise ValueError('Self-loop is not allowed.')
        rv = self._graph.add_edge(from_node, to_node)
        validate_multi_edges(from_node)
        validate_multi_edges(to_node)
        return rv

    def _filter_nodes(self, func):
        return (n for n in self.nodes() if func(n))

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

    def inputs(self):
        return self._filter_nodes(lambda n: not tuple(self.predecessors(n)))

    def outputs(self):
        return self._filter_nodes(lambda n: not tuple(self.successors(n)))

    def tensor_nodes(self):
        return self._filter_nodes(lambda n: isinstance(n, TensorNode))

    def layer_nodes(self):
        return self._filter_nodes(lambda n: isinstance(n, LayerNode))

    def topological_order(self):
        return nx.topological_sort(self)

    def _add_module(
            self, from_names, to_names,
            module_name, module_params, module_path=None):
        # replace kwargs in module params
        params = _replace_module_kwargs(module_params)
        # module path
        module_path = module_path or []
        submodule_path = list(module_path)
        if module_name is not None:
            submodule_path += [module_name]
        # add graph connections
        for connection in params['graph']:
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
        input_names = params.get('inputs', ['input'])
        for from_name, input_name in zip(from_names, input_names):
            from_node = TensorNode(from_name, module_path)
            input_node = TensorNode(input_name, submodule_path)
            self.add_edge(from_node, input_node)
        output_names = params.get('outputs', ['output'])
        for output_name, to_name in zip(output_names, to_names):
            output_node = TensorNode(output_name, submodule_path)
            to_node = TensorNode(to_name, module_path)
            self.add_edge(output_node, to_node)

    def _add_layer(
            self, from_names, to_names,
            layer_name, layer_params, module_path):
        from_names = ensure_list(from_names)
        to_names = ensure_list(to_names)
        if layer_params is not None and layer_params['type'] == 'module':
            return
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
        if layer_params is None:
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
        # IO is unique
        inputs = list(self.inputs())
        outputs = list(self.outputs())
        io_error = not (len(inputs) == len(outputs) == 1)
        io_error = io_error or (inputs[0].name != 'images')
        io_error = io_error or (outputs[0].name != 'logits')
        if io_error:
            raise GraphIOError(
                'We expect the net to have a unique input "images" and '
                'a unique output "logits".')
        # graph is connected
        if not any(nx.all_simple_paths(self._graph, inputs[0], outputs[0])):
            raise GraphIOError(
                'We expect the net to have a path from the input '
                '"images" to the output "logits".')

    def instantiate(self, iodef):
        def assign_value(node, pred):
            if node.value is not None:
                raise ValueError(
                    'Node {!r} has already been assigned a value.'
                    .format(node))
            if isinstance(pred, collections.Sequence):
                node.value = [n.value for n in pred]
                is_none = any(v is None for v in node.value)
            else:
                node.value = pred.value
                is_none = node.value is None
            if is_none:
                raise ValueError(
                    'Node {!r} is not fully assigned.'.format(pred))

        for each_node in self.topological_order():
            pred_nodes = tuple(self.predecessors(each_node))
            if isinstance(each_node, JoinNode):
                assign_value(each_node, pred_nodes)
            elif isinstance(each_node, SplitNode):
                if len(pred_nodes) > 1:
                    raise IndexError(
                        'Number of predecessors of {!r} must be 1.'
                        .format(each_node))
                assign_value(each_node, pred_nodes[0])
            elif isinstance(each_node, TensorNode):
                # TODO SplitNode handling
                assign_value(each_node, pred_nodes[0])
            elif isinstance(each_node, LayerNode):
                # TODO instantiate layer
                ...
            else:
                raise TypeError(
                    'Node {!r} has an unrecognized type.'.format(each_node))


if __name__ == "__main__":
    from mayo.config import Config
    config = Config()
    config.yaml_update('models/squeezenet_v11.yaml')
    model = config.model
    net = NetBase(model.name, model.layers, model.graph)
    __import__('ipdb').set_trace()
