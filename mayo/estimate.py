import time
import collections

import numpy as np

from mayo.log import log
from mayo.util import object_from_params, Change
from mayo.net.graph import LayerNode, TensorNode, SplitNode


class ResourceEstimator(object):
    default_history = 100

    def __init__(self, batch_size, allow_reregister=True):
        super().__init__()
        self.batch_size = batch_size
        self.allow_reregister = allow_reregister
        self.change = Change()
        self.operations = {}
        self.statistics = {}
        self.properties = {}
        self.formatters = []
        self.shapes = None

    def __getstate__(self):
        return {
            'statistics': self.statistics,
            'properties': self.properties,
        }

    def register(
            self, tensor, name, node=None, history=None,
            transformer=None, formatter=None):
        """
        Register statistic tensors to be run by session.

        tensor: the value tensor to be run.
        name: name of the given statistic.
        node:
            the associated layer node, if not specified, it defaults to
            'global'.
        history:
            the number of past values to keep, if history='infinite' we do not
            discard past values; if not specified, we keep 100.
        transformer: transform value before adding to statistics.
        formatter: calls .register_print with `formatter`.
        """
        history = self.default_history if history is None else history
        node = 'global' if node is None else node
        layer = self.operations.setdefault(node, {})
        prop = self.properties.setdefault(node, {})
        if name in layer and not self.allow_reregister:
            raise ValueError(
                'Tensor named {!r} already registered for layer {!r}.'
                .format(name, layer))
        layer[name] = tensor
        prop[name] = {'history': history, 'transformer': transformer}
        if formatter:
            self.register_formatter(formatter)

    def register_formatter(self, func):
        """
        Register function to call for print formatting.

        func:
            a function which accepts the estimator instance and returns (name,
            value) for print formatting; if it is not specified, this statistic
            will not be printed.
        """
        if func not in self.formatters:
            self.formatters.append(func)

    def add(self, value, name, node=None):
        node = node or 'global'
        try:
            history = self.properties[node][name]['history']
        except KeyError:
            history = self.default_history
        stats = self.statistics.setdefault(node, {})
        values = stats.setdefault(name, [])
        if history != 'infinite':
            while len(values) >= history:
                values.pop(0)
        values.append(value)

    def append(self, statistics):
        """
        Add new statistics to the estimator instance.

        statistics: a [layer_node][statistic_name]-value nested mapping.
        """
        for layer, stats in statistics.items():
            curr_stats = self.statistics.setdefault(layer, {})
            prop = self.properties[layer]
            for key, value in stats.items():
                values = curr_stats.setdefault(key, [])
                history = prop[key]['history']
                if history != 'infinite':
                    while len(values) >= history:
                        values.pop(0)
                transformer = prop[key]['transformer']
                if transformer:
                    value = transformer(value)
                values.append(value)

    def max_len(self, name=None):
        l = 0
        for stats in self.statistics.values():
            for stat_name, history in stats.items():
                if name is not None and name != stat_name:
                    continue
                l = max(l, len(history))
        return l

    def format(self, batch_size=None):
        text = []
        for func in self.formatters:
            text.append(func(self))
        if batch_size:
            # performance
            interval = self.change.delta('step.duration', time.time())
            if interval != 0:
                imgs_per_sec = batch_size / float(interval)
                imgs_per_sec = self.change.moving_metrics(
                    'step.imgs_per_sec', imgs_per_sec, std=False)
                text.append('tp: {:4.0f}/s'.format(imgs_per_sec))
        return ' | '.join(text)

    def get_history(self, name, node=None):
        return self.statistics[node or 'global'][name]

    def flush(self, name, node=None):
        del self.statistics[node or 'global'][name]

    def flush_all(self, name):
        for node, stats in self.statistics.items():
            for stat_name in list(stats):
                if stat_name == name:
                    del stats[name]

    def get_histories(self, name):
        return {
            layer_name: layer_stats[name]
            for layer_name, layer_stats in self.statistics.items()
            if name in layer_stats
        }

    def set_history(self, value, name, node=None):
        self.statistics[node or 'global'][name] = value

    def get_value(self, name, node=None):
        return self.get_history(name, node)[-1]

    def get_values(self, name):
        return {
            layer_name: values[-1] if len(values) else None
            for layer_name, values in self.get_histories(name).items()
        }

    def get_mean(self, name, node=None):
        history = self.get_history(name, node)
        return np.mean(history)

    def get_mean_std(self, name, node=None):
        history = self.get_history(name, node)
        return np.mean(history), np.std(history)

    def get_tensor(self, name, node=None):
        return self.operations[node or 'global'][name]

    def add_estimate(self):
        if not self.shapes:
            raise ValueError('Shape of nodes is not set.')
        for node, shape in self.shapes.items():
            if not isinstance(node, LayerNode):
                continue
            layer = self.statistics.setdefault(node, {})
            try:
                func, params = object_from_params(
                    node.params, self, 'estimate_')
            except NotImplementedError:
                continue
            # tensors
            inputs = [self.shapes[p] for p in node.predecessors]
            inputs = inputs[0] if len(inputs) == 1 else inputs
            stats = func(node, inputs, shape, params)
            stats = {name: [value] for name, value in stats.items()}
            layer.update(stats)

    @staticmethod
    def _multiply(items):
        value = 1
        for i in items:
            value *= i
        return value

    @classmethod
    def _kernel_size(cls, node):
        kernel = node.params['kernel_size']
        if isinstance(kernel, collections.Sequence):
            return cls._multiply(kernel)
        elif isinstance(kernel, int):
            return kernel * kernel
        raise TypeError(
            'We do not understand the kernel size {!r}.'.format(kernel))

    def estimate_depthwise_convolution(
            self, node, input_shape, output_shape, params):
        # output feature map size (H x W x C_out)
        macs = list(output_shape[1:])
        # kernel size K_H x K_W
        macs.append(self._kernel_size(node))
        return {'MACs': self._multiply(macs)}

    def estimate_convolution(self, node, input_shape, output_shape, params):
        depthwise_stats = self.estimate_depthwise_convolution(
            node, input_shape, output_shape, params)
        # input channel size C_in
        macs = depthwise_stats['MACs'] * int(input_shape[-1])
        return {'MACs': macs}

    def estimate_fully_connected(
            self, node, input_shape, output_shape, params):
        return {'MACs': int(input_shape[-1] * output_shape[-1])}

    def _gate_density(self, gates):
        if not gates:
            log.warn(
                'Cannot estimate gate density without a history of gates, '
                'defaulting to zero sparsity.')
            return 1
        valids = sum(np.sum(g.astype(np.int32)) for g in gates)
        totals = sum(g.size for g in gates)
        return valids / totals

    def _gate_for_node(self, node):
        """
        Recursively find the gate sparsity of the predecessor nodes
        that can propagate sparsity.
        """
        def true():
            return [np.ones(self.shapes[node], dtype=bool)] * self.max_len()

        if isinstance(node, SplitNode):
            return self._gate_for_node(node.predecessors[0])
        if isinstance(node, TensorNode):
            preds = node.predecessors
            if preds:
                return self._gate_for_node(preds[0])
            return true()
        if not isinstance(node, LayerNode):
            raise TypeError(
                'Do not know how to find gate sparsity for node {!r}'
                .format(node))
        ntype = node.params['type']
        if ntype == 'gated_convolution':
            if node.params.get('should_gate', True):
                try:
                    return self.get_history('gate.active', node)
                except KeyError:
                    log.warn(
                        'No gate history available, defaulting to '
                        'zero sparsity.')
            return true()
        passthrough_types = [
            'dropout', 'max_pool', 'average_pool', 'activation', 'identity']
        if ntype in passthrough_types:
            return self._gate_for_node(node.predecessors[0])
        if ntype in ['concat', 'add', 'mul']:
            histories = [
                self._gate_for_node(p)
                for p in node.predecessors[0].predecessors]
            history = []
            for each in zip(*histories):
                if ntype == 'concat':
                    each = np.concatenate(each, axis=-1)
                else:
                    func = np.add if ntype == 'add' else np.multiply
                    result = None
                    for h in each:
                        h = h.astype(int)
                        if result is None:
                            result = h
                        else:
                            result = func(result, h)
                    each = result.astype(bool)
                history.append(each)
            return history
        return true()

    def estimate_gated_convolution(
            self, node, input_shape, output_shape, params):
        out_density = self._gate_density(self._gate_for_node(node))
        # gated convolution expects only one input
        in_node = node.predecessors[0]
        in_density = self._gate_density(self._gate_for_node(in_node))
        stats = self.estimate_convolution(
            node, input_shape, output_shape, params)
        stats['MACs'] = int(stats['MACs'] * in_density * out_density)
        # gating network overhead
        # io channels
        overhead = int(input_shape[-1] * output_shape[-1])
        granularity = params.get('granularity', 'channel')
        if granularity == 'vector':
            kernel_height = params['kernel_size']
            if not isinstance(kernel_height, int):
                kernel_height, _ = kernel_height
            # vector-wise gating output shape: height
            # and input kernel height
            overhead *= output_shape[1] * kernel_height
        # TODO parametric gamma element-wise multiply + add overhead
        stats['MACs'] += overhead
        return stats
