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
        regularizer = self.regularizer
        if isinstance(regularizer, str):
            regularizer = [regularizer]
        sparse = self.gate() * self.actives()
        loss = []
        if 'l1' in self.regularizer:
            loss.append(tf.abs(sparse))
        if 'l2' in self.regularizer:
            loss.append(tf.square(sparse))
        if 'moe' in self.regularizer:
            # mixture of experts
            loss.append(self._mixture(sparse, [0, 1, 2]))
        if 'moi' in self.regularizer:
            # mixture of idiots
            loss.append(self._mixture(sparse, [1, 2, 3]))
        loss = tf.add_n([tf.reduce_sum(l) for l in loss])
        self._add_regularization(loss)
