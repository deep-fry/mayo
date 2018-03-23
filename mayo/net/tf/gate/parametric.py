import tensorflow as tf
from tensorflow.contrib import slim

from mayo.util import memoize_method
from mayo.net.tf.gate.base import (
    SparseRegularizedGatedConvolutionBase, GateParameterValueError)


class ParametricGatedConvolution(SparseRegularizedGatedConvolutionBase):
    """ Parametric batch normalization with gating.  """
    def _update_defaults(self, defaults):
        super()._update_defaults(defaults)
        # FIXME hacky normalizer customization
        defaults['norm'] = 'batch'
        defaults['parametric_beta'] = False

    def normalize(self, tensor):
        if self.normalizer_fn is not slim.batch_norm:
            raise GateParameterValueError(
                'Policy "{}" is used, we expect slim.batch_norm to '
                'be used but it is absent in {}.'
                .format(self.policy, self.node))
        if not self.normalizer_params.get('scale', False):
            raise GateParameterValueError(
                'Policy "parametric_gamma" expects `scale` to be used '
                'in slim.batch_norm.')
        if self.norm == 'batch':
            normalizer_params = dict(self.normalizer_params, **{
                'scale': False,
                'center': False,
                'activation_fn': None,
                'scope': '{}/BatchNorm'.format(self.scope),
                'is_training': self.is_training,
            })
            return self.constructor.instantiate_batch_normalization(
                None, tensor, normalizer_params)
        if self.norm == 'channel':
            norm_mean, norm_var = tf.nn.moments(
                tensor, axes=[1, 2], keep_dims=True)
            return (tensor - norm_mean) / tf.sqrt(norm_var)
        raise GateParameterValueError('Unrecognized normalization policy.')

    @memoize_method
    def beta(self):
        tensor = self._predictor('gate/beta')
        self._register('beta', tensor)
        return tensor

    def activate(self, tensor):
        # gating happens before activation
        # output = relu(
        #   actives(gamma(x)) * gamma(x) * norm(conv(x)) +
        #   actives(gamma(x)) * beta
        # )
        actives = self.actives()
        gamma = self.gate()
        tensor *= actives * gamma if self.enable else gamma
        if self.normalizer_params.get('center', True):
            if self.parametric_beta:
                beta = self.beta()
            else:
                # constant beta
                beta_scope = '{}/gate/shift'.format(self.scope)
                beta = tf.get_variable(
                    beta_scope, shape=tensor.shape[-1], dtype=tf.float32,
                    initializer=tf.constant_initializer(0.1),
                    trainable=self.trainable)
            tensor += actives * beta if self.enable else beta
        return super().activate(tensor)
