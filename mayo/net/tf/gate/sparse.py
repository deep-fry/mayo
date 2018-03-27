import tensorflow as tf

from mayo.net.tf.gate.base import GatedConvolutionBase


class SparseRegularizedGatedConvolutionBase(GatedConvolutionBase):
    def _update_defaults(self, defaults):
        defaults['regularizer'] = 'l1'
        defaults['epsilon'] = 0.001

    def _mixture(self, tensor, axes):
        tensor = tf.reduce_sum(tensor, axes)
        if tensor.shape.ndims != 1:
            raise ValueError('Tensor should reduce to a vector.')
        mean, variance = tf.nn.moments(tensor, axes=0)
        return variance / tf.square((mean + self.epsilon))

    def regularize(self):
        """
        We use a L1, L2 or MoE regularizer to encourage sparsity in gate.
        """
        func_map = {
            'l1': lambda v: tf.abs(v),
            'l2': lambda v: tf.square(v),
            'moe': lambda v: self._mixture(v, [0, 1, 2]),
            'moi': lambda v: self._mixture(v, [1, 2, 3]),
        }
        # set inactive elements to zeros
        # Xitong: it would be possible to use self.actives() to mask out
        # elements to regularize, but writing this in tensorflow has
        # proven to be tricky.
        sparse = self.gate() * self.actives()
        # add regularization for each specified keys
        for key, weight in self.regularizer.items():
            self._add_regularization(func_map[key](sparse), weight)
