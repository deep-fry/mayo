import math

import numpy as np

from mayo.log import log
from mayo.util import Percent, memoize_method
from mayo.net.tf.gate.base import GateError
from mayo.net.tf.gate.naive import NaiveGatedConvolution
from mayo.net.tf.gate.squeeze import SqueezeExciteGatedConvolution
from mayo.net.tf.gate.parametric import ParametricGatedConvolution


class GatePolicyTypeError(GateError):
    """Unrecognized policy.  """


class GateLayers(object):
    """Layer implementations for gated convolution.  """

    @staticmethod
    def _gate_loss_formatter(estimator):
        # gating loss for printing
        losses = estimator.get_histories('gate.loss')
        total_losses = None
        for loss_history in losses.values():
            if total_losses is None:
                total_losses = list(loss_history)
            else:
                total_losses = [
                    a + b for a, b in zip(total_losses, loss_history)]
        if total_losses is None:
            loss_mean = 0
        else:
            loss_mean = np.mean(total_losses)
        if loss_mean > 0:
            loss_std = Percent(np.std(total_losses) / loss_mean)
        else:
            loss_std = '?%'
        if math.isnan(loss_mean):
            log.error(
                'Gating loss is NaN. Please check your regularizer weight.')
        return 'gate.loss: {:.5f}±{}'.format(loss_mean, loss_std)

    @staticmethod
    def _gate_density_formatter(estimator):
        gates = estimator.get_values('gate.active')
        if not gates:
            return 'gate: off'
        valid = total = 0
        for layer, gate in gates.items():
            valid += np.sum(gate.astype(np.float32) != 0)
            total += gate.size
        return 'gate: {}'.format(Percent(valid / total))

    @memoize_method
    def _register_gate_formatters(self):
        self.session.estimator.register_formatter(self._gate_loss_formatter)
        self.session.estimator.register_formatter(self._gate_density_formatter)

    _policy_map = {
        'naive': NaiveGatedConvolution,
        'parametric': ParametricGatedConvolution,
        'squeeze': SqueezeExciteGatedConvolution,
    }

    def instantiate_gated_convolution(self, node, tensor, params):
        # register gate sparsity for printing
        self._register_gate_formatters()
        # params
        gate_params = params.pop('gate_params')
        policy = gate_params.pop('policy')
        try:
            cls = self._policy_map[policy]
        except KeyError:
            raise GatePolicyTypeError('Unrecognized gated convolution policy.')
        return cls(self, node, params, gate_params, tensor).instantiate()

    def _estimate_overhead(self, info, input_shape, output_shape):
        in_channels = input_shape[-1]
        out_channels = output_shape[-1]
        if self.factor <= 0:
            return in_channels * out_channels
        mid_channels = math.ceil(self.num_outputs / self.factor)
        macs = in_channels * mid_channels
        macs += mid_channels * out_channels
        return macs

    def estimate_gated_convolution(
            self, node, info, input_shape, output_shape, params):
        layer_info = self.constructor.estimate_convolution(
            info, input_shape, output_shape, params)
        if self.enable:
            mask = self.estimator.get_history('gate.active', info['_node'])
            density = self.estimator._mask_density(mask)
            layer_info['_mask'] = mask
            layer_info['density'] = density
            layer_info['macs'] *= density
        overhead = self._estimate_overhead(info, input_shape, output_shape)
        layer_info['overhead'] = overhead
        layer_info['macs'] += overhead
        return layer_info
