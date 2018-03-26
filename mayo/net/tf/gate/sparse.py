import tensorflow as tf

from mayo.net.tf.gate.base import GatedConvolutionBase


class SparseRegularizedGatedConvolutionBase(GatedConvolutionBase):
    def _update_defaults(self, defaults):
        defaults['regularizer'] = 'l1'
        defaults['epsilon'] = 0.001

    def _mixture(self, tensor, axes):
        mean, variance = tf.nn.moments(tensor, axes=axes)
        return variance / tf.square((mean + self.epsilon))

    def regularize(self):
        """
        We use a L1, L2 or MoE regularizer to encourage sparsity in gate.
        """
        sparse = self.gate() * self.actives()
        func_map = {
            'l1': lambda v: tf.abs(v),
            'l2': lambda v: tf.square(v),
            'moe': lambda v: self._mixture(v, [0, 1, 2]),
            'moi': lambda v: self._mixture(v, [1, 2, 3]),
        }
        for key, weight in self.regularizer.items():
            self._add_regularization(func_map[key](sparse), weight)
