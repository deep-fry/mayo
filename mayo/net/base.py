import networkx as nx

from mayo.util import ensure_list


class NetBase(nx.OrderedMultiDiGraph):
    def __init__(self, layer_defs, graph_defs):
        super().__init__()
        self._add_module(layer_defs, ensure_list(graph_defs), [])

    def _filter_nodes(self, func):
        return (n for n in self.nodes() if func(n))

    def inputs(self):
        return self._filter_nodes(lambda n: not tuple(self.predecessors(n)))

    def outputs(self):
        return self._filter_nodes(lambda n: not tuple(self.successors(n)))

    def tensor_nodes(self):
        return self._filter_nodes(lambda n: n['type'] == 'tensor')

    def layer_nodes(self):
        return self._filter_nodes(lambda n: n['type'] == 'layer')

    def _node(self, name, node_type, module_path, params=None):
        node = {
            'type': node_type,
            'name': name,
            'module': module_path,
        }
        if params is not None:
            node['params'] = params
        return node

    def _add_module(self, layer_defs, graph_defs, module_path):
        for connection in graph_defs:
            from_nodes = ensure_list(connection['from'])
            to_nodes = ensure_list(connection['to'])
            with_layers = ensure_list(connection['with'])
            paths = zip(
                [from_nodes] + with_layers,
                with_layers + [to_nodes],
                with_layers + [None])
            for input_nodes, output_nodes, layer in paths:
                self._add_layer(input_nodes, output_nodes, layer, module_path)

    def _add_layer(self, from_nodes, to_nodes, layer, module_path):
        if len(from_nodes) == 1:
            join_node = from_nodes[0]
        else:
            # join input nodes
            join_name = ', '.format(from_nodes)
            join_node = self._node(join_name, 'join', module_path)
            for each in from_nodes:
                self.add_edge(self._node(each, 'tensor'), join_node)
        # layer
        if layer is None:
            layer_node = join_node
        else:
            params = self.layer_attributes[layer]
            if params['type'] == 'module':
                module_path = module_path + [params['name']]
                self._add_module(
                    params['layers'], params['graph'], module_path)
                # TODO add IO interface to module
                layer_node = ...
            else:
                layer_node = self._node(layer, 'layer', module_path, params)
            self.add_edge(join_node, layer_node)
        if len(to_nodes) == 1:
            self.add_edge(layer_node, to_nodes[0])
        else:
            split_node = self._node(', '.join(to_nodes), 'split', module_path)
            self.add_edge(layer, split_node)
            for each in to_nodes:
                each = {'type': 'tensor', 'name': each}
                self.add_edge(split_node, each)

    def instantiate(self, iodef):
        def assign_value(node, value):
            try:
                value = node['value']
            except KeyError:
                pass
            else:
                raise ValueError(
                    'Node {!r} has already been assigned a value {!r}'
                    .format(node, value))
            node['value'] = value

        nodes = nx.topological_sort(self)
        for each in nodes:
            if each['type'] == 'join':
                assign_value(
                    each, tuple(n['value'] for n in self.predecessors(each)))
            elif each['type'] in ['split', 'tensor']:
                preds = tuple(self.predecessors(each))
                if len(preds) > 1:
                    raise IndexError(
                        'Number of predecessors of a {!r} node must be 1.'
                        .format(each['type']))
                assign_value(each, preds[0])
            elif each['type'] == 'layer':
                ...
            else:
                raise TypeError(
                    'Unrecognized node type {!r}.'.format(each['type']))
