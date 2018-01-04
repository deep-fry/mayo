import time
import collections

import numpy as np
import tensorflow as tf

from mayo.util import object_from_params, Change
from mayo.net.graph import LayerNode


class ResourceEstimator(object):
    def __init__(self):
        super().__init__()
        self.change = Change()
        self.operations = {}
        self.statistics = {}
        self.properties = {}
        self.formatters = []

    def register(self, tensor, name, node=None, history=None, formatter=None):
        """
        Register statistic tensors to be run by session.

        tensor: the value tensor to be run.
        name: name of the given statistic.
        node:
            the associated layer node, if not specified, it defaults to
            'global'.
        history:
            the number of past values to keep, if history='infinite' we do not
            discard past values; if not specified, we keep only one.
        formatter: calls .register_print with `formatter`.
        """
        history = 100 if history is None else history
        node = 'global' if node is None else node
        layer = self.operations.setdefault(node, {})
        prop = self.properties.setdefault(node, {})
        if name in layer:
            raise ValueError(
                'Tensor named {!r} already registered for layer {!r}.'
                .format(name, layer))
        layer[name] = tensor
        prop[name] = {'history': history}
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
        self.formatters.append(func)

    def add(self, statistics):
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
                values.append(value)

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
            name: values[-1]
            for name, values in self.get_histories(name).items()
        }

    def get_mean(self, name, node=None):
        history = self.get_history(name, node)
        return np.mean(history)

    def get_mean_std(self, name, node=None):
        history = self.get_history(name, node)
        return np.mean(history), np.std(history)

    def get_tensor(self, name, node=None):
        return self.operations[node or 'global'][name]

    def add_estimate(self, layer_tensors):
        for n, tensor in layer_tensors.items():
            if not isinstance(n, LayerNode):
                continue
            layer = self.statistics.setdefault(n, {})
            try:
                func, params = object_from_params(n.params, self, 'estimate_')
            except NotImplementedError:
                continue
            # tensors
            inputs = [layer_tensors[p] for p in n.predecessors]
            inputs = inputs[0] if len(inputs) == 1 else inputs
            stats = func(n, inputs, tensor, params)
            stats = {name: [value] for name, value in stats.items()}
            layer.update(stats)

    @staticmethod
    def _multiply(items):
        value = 1
        for i in items:
            if isinstance(i, tf.Dimension):
                i = int(i)
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
            self, node, input_tensor, output_tensor, params):
        # output feature map size (H x W x C_out)
        macs = list(output_tensor.shape[1:])
        # kernel size K_H x K_W
        macs.append(self._kernel_size(node))
        return {'MACs': self._multiply(macs)}

    def estimate_convolution(self, node, input_tensor, output_tensor, params):
        depthwise_stats = self.estimate_depthwise_convolution(
            node, input_tensor, output_tensor, params)
        # input channel size C_in
        macs = depthwise_stats['MACs'] * int(input_tensor.shape[-1])
        return {'MACs': macs}

    def estimate_fully_connected(
            self, node, input_tensor, output_tensor, params):
        return {'MACs': int(input_tensor.shape[-1] * output_tensor.shape[-1])}

    def _gated_density(self, node):
        if not isinstance(node, LayerNode):
            return 1
        if node.params['type'] != 'gated_convolution':
            return 1
        try:
            gates = self.get_history('gate', node)
        except KeyError:
            return 1
        if not gates:
            return 1
        valids = sum(np.sum(g.astype(np.int32)) for g in gates)
        totals = sum(g.size for g in gates)
        return valids / totals

    def estimate_gated_convolution(
            self, node, input_tensor, output_tensor, params):
        out_density = self._gated_density(node)
        preds = node.predecessors
        if len(preds) > 1:
            raise ValueError(
                'Number of predecessors should only be 1 for '
                'gated_convolution.')
        if not preds:
            in_density = input_tensor.shape[-1]
        else:
            in_density = self._gated_density(preds[0])
        stats = self.estimate_convolution(
            node, input_tensor, output_tensor, params)
        stats['MACs'] *= in_density * out_density
        # gating network overhead
        stats['MACs'] += int(input_tensor.shape[-1] * output_tensor.shape[-1])
        stats['MACs'] = int(stats['MACs'])
        return stats
