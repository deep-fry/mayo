import time
import functools
import collections

import numpy as np
import tensorflow as tf

from mayo.log import log
from mayo.util import object_from_params, Change
from mayo.net.graph import LayerNode


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
        self.debuggers = []
        self.net = None

    def __getstate__(self):
        return {
            'statistics': self.statistics,
            'properties': self.properties,
        }

    def register(
            self, tensor, name, node=None, history=None,
            transformer=None, formatter=None, debugger=None):
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
        debugger: function to print extra debug info.
        """
        if not isinstance(tensor, (tf.Tensor, tf.Variable)):
            raise TypeError('We expect {!r} to be a Tensor'.format(tensor))
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
        if debugger:
            self.register_debugger(debugger)

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

    def register_debugger(self, func):
        if func not in self.debuggers:
            self.debuggers.append(func)

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

    def debug(self):
        for func in self.debuggers:
            log.debug(
                'Estimator debug info for {!r}: {}'
                .format(func.__qualname__, func(self)))

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

    def get_tensors(self, name):
        return {
            layer_name: stat_tensors[name]
            for layer_name, stat_tensors in self.operations.items()
            if name in stat_tensors
        }

    @staticmethod
    def _multiply(items):
        value = 1
        for i in items:
            value *= i
        return value

    @classmethod
    def _kernel_size(cls, params):
        kernel = params['kernel_size']
        if isinstance(kernel, collections.Sequence):
            return cls._multiply(kernel)
        elif isinstance(kernel, int):
            return kernel * kernel
        raise TypeError(
            'We do not understand the kernel size {!r}.'.format(kernel))

    @staticmethod
    def _mask_density(mask):
        if not mask:
            return 1
        valids = sum(np.sum(m.astype(np.int32)) for m in mask)
        totals = sum(m.size for m in mask)
        return float(valids / totals)

    @staticmethod
    def _mask_join(masks, reducer):
        combined_mask = []
        i = 0
        while True:
            try:
                each = [m[i] if isinstance(m, list) else m for m in masks]
            except IndexError:
                break
            i += 1
        for each in zip(*masks):
            combined_mask.append(functools.reduce(reducer, each))
        return combined_mask

    @staticmethod
    def _mask_passthrough(info, layer_info):
        if 'density' in info:
            layer_info['density'] = info['density']
        if '_mask' in info:
            layer_info['_mask'] = info['_mask']
        return layer_info

    @staticmethod
    def _apply_input_sparsity(info, layer_info):
        if '_mask' not in info:
            return layer_info
        macs = layer_info['macs']
        layer_info['macs'] = macs * info.get('density', 1)
        layer_info['_original_macs'] = macs
        return layer_info
