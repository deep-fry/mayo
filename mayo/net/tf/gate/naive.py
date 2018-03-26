import tensorflow as tf

from mayo.net.tf.gate.base import (
    GatedConvolutionBase, GateParameterValueError)


class NaiveGatedConvolution(GatedConvolutionBase):
    def _update_defaults(self, defaults):
        defaults['regularizer'] = 'mse'

    def instantiate(self):
        output = super().instantiate()
        return self.actives() * output

    def regularize(self):
        """
        Regularize gate by making gate output `gate` to match `match` as close
        as possible.
        """
        gate = self.gate()
        match = tf.stop_gradient(self.subsample(self.activated))
        # policy descriminator: we simply match values in each channel
        # using a loss regularizer
        loss = None
        if 'mse' in self.regularizer:
            loss = tf.losses.mean_squared_error(
                match, gate, loss_collection=None)
            self._add_regularization(loss, self.regularizer['mse'])
        if 'sce' in self.regularizer:
            loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=match, logits=gate)
            self._add_regularization(loss, self.regularizer['sce'])
        if loss is None:
            raise GateParameterValueError(
                'We expect weight to be non-zero if `match` is specified, '
                'as without `match` to regularize gate, the gate network '
                'is not learning anything.')
