import collections

import tensorflow as tf  # FIXME estimator is tensorflow-only for now

from mayo.util import object_from_params
from mayo.net.graph import LayerNode


class ResourceEstimator(object):
    def __init__(self, net):
        super().__init__()
        self.statistics = self._estimate(net)

    def _estimate(self, net):
        statistics = {}
        for n in net._graph.topological_order():
            if not isinstance(n, LayerNode):
                continue
            try:
                func, params = object_from_params(n.params, self, 'estimate_')
            except NotImplementedError:
                statistics[n] = None
                continue
            # tensors
            inputs = [net._tensors[p] for p in n.predecessors]
            if len(inputs) == 1:
                inputs = inputs[0]
            outputs = net._tensors[n]
            statistics[n] = func(n, inputs, outputs, params)
        total = {}
        for n, stat in statistics.items():
            if not stat:
                continue
            for k, v in stat.items():
                total[k] = total.get(k, 0) + v
        statistics['total'] = total
        return statistics

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
        return {'MACs': input_tensor.shape[-1] * params['num_outputs']}
