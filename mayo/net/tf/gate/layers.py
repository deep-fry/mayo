import math
import functools

import numpy as np

from mayo.log import log
from mayo.util import Percent, memoize_method
from mayo.net.tf.estimate import multiply, mask_density
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

    def _estimate_overhead(
            self, in_shape, out_shape, in_density, active_density, params):
        in_channels = int(in_shape[-1] * in_density)
        out_channels = int(out_shape[-1] * active_density)
        factor = params.get('factor', 0)
        if factor <= 0:
            macs = in_channels * out_channels
            # FC uses number of weights = (MACs + bias parameters)
            weights = macs + out_channels
        else:
            mid_channels = math.ceil(params['num_outputs'] / factor)
            macs = in_channels * mid_channels
            macs += mid_channels * out_channels
            weights = NotImplemented
        # gamma multiplication overhead
        macs += multiply(out_shape[1:])
        return weights, macs

    def estimate_gated_convolution(
            self, node, in_info, in_shape, out_shape, params):
        out_info = self.estimate_convolution(
            node, in_info, in_shape, out_shape, params)
        active_density = 1
        if params.get('enable', True):
            try:
                mask = self.estimator.get_history('gate.active', node)
            except KeyError:
                pass
            else:
                density, active_density = mask_density(mask)
                out_info['_mask'] = mask
                out_info['active'] = active_density
                out_info['density'] = density
                out_info['macs'] = int(out_info['macs'] * density)
                out_info['weights'] = int(out_info['weights'] * active_density)
        in_density = in_info.get('density', 1)
        oweights, omacs = self._estimate_overhead(
            in_shape, out_shape, in_density, active_density, params)
        # out_info['overhead'] = overhead_macs
        out_info['weights'] += oweights
        out_info['macs'] += omacs
        return out_info
