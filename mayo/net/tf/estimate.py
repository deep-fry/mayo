import math
import functools
import collections

import numpy as np

from mayo.util import Percent, Bits
from mayo.override.base import ChainOverrider
from mayo.override.prune.base import PrunerBase
from mayo.override.quantize.base import QuantizerBase


def multiply(items):
    value = 1
    for i in items:
        value *= i
    return value


def _kernel_size(params):
    kernel = params['kernel_size']
    if isinstance(kernel, collections.Sequence):
        return multiply(kernel)
    elif isinstance(kernel, int):
        return kernel * kernel
    raise TypeError(
        'We do not understand the kernel size {!r}.'.format(kernel))


def mask_density(mask):
    if not mask:
        return 1, 1
    # mask
    valids = sum(np.sum(m.astype(np.int32)) for m in mask)
    totals = sum(m.size for m in mask)
    density = Percent(valids / totals)
    # active
    for mm in mask:
        if mm.ndim == 1:
            # channel pruning, static mask
            active = mm
            break
    else:
        flat_masks = (m for mm in mask for m in mm)
        active = functools.reduce(np.logical_or, flat_masks)
    active = Percent(np.sum(active) / active.size)
    return density, active


def mask_join(masks, reducer):
    length = 1
    for hist in masks:
        if isinstance(hist, list):
            length = max(length, len(hist))
    masks = [
        [hist] * length if not isinstance(hist, list) else hist
        for hist in masks]
    return [functools.reduce(reducer, each) for each in zip(*masks)]


def passthrough(in_info, out_info):
    if 'density' in in_info:
        out_info['density'] = in_info['density']
    if 'active' in in_info:
        out_info['active'] = in_info['active']
    if '_mask' in in_info:
        out_info['_mask'] = in_info['_mask']
    return out_info


def _adder_tree(inputs, width):
    float_add_binops = None
    adders = height = binops = 0
    while inputs != 1:
        inputs = int(math.ceil(inputs / 2))
        adders += inputs
        height += 1
        if width != 'float':
            width += 1
            binops += inputs * width
        else:
            binops += float_add_binops
    return {'adders': adders, 'height': height, 'binops': binops}


def apply_sparsity(
        weight_overrider, in_info, out_info, in_shape, out_shape,
        depthwise=False):
    num_inputs = multiply(in_shape[1:])
    num_outputs = multiply(out_shape[1:])
    weight_bitwidth = 32
    weight_density = 1.0
    if weight_overrider:
        if isinstance(weight_overrider, ChainOverrider):
            overriders = weight_overrider
        else:
            overriders = [weight_overrider]
        for o in overriders:
            if isinstance(o, PrunerBase):
                weight_density = o.info().density
            if isinstance(o, QuantizerBase):
                if hasattr(o.info(), "real_width"):
                    weight_bitwidth = o.info().real_width
    in_density = in_info.get('density', 1.0)
    in_bitwidth = in_info.get('bitwidth', 32)
    out_density = out_info.get('density', 1.0)
    out_bitwidth = out_info.get('bitwidth', 32)
    mem_input = int(num_inputs * in_density * in_bitwidth)
    mem_output = int(num_outputs * out_density * out_bitwidth)
    if depthwise:
        full_density = out_density
        mem_activation = max(mem_input, mem_output)  # inplace computation
    else:
        full_density = in_density * out_density
        mem_activation = mem_input + mem_output
    weights = out_info['weights'] * weight_density
    mem_weights = int(weights * full_density * weight_bitwidth)
    active_density = out_info.get('active', 1)
    if not depthwise:
        active_density *= in_info.get('active', 1)
    macs = int(out_info['macs'] * full_density)
    update_info = {
        'macs': macs,
        'weights': int(weights * active_density),
        'mem_weights': Bits(mem_weights),
        'mem_activation': Bits(mem_activation),
        # 'alu_moves': int(macs * 2 + num_outputs * out_density),
        # 'optimal_cache': Bits(mem_weights + mem_activation),
        # TODO fixed point bitwidth after multiplication
        # 'binops': _adder_tree(
        #     num_inputs * in_density * weight_density, 0)['binops'],
    }
    return dict(out_info, **update_info)


class LayerEstimateMixin(object):
    def _estimate_depthwise_convolution(self, out_shape, params):
        # kernel size K_h x K_w
        kernel = _kernel_size(params)
        # weights, K_h x K_w x C_out
        weights = multiply([kernel, out_shape[-1]])
        # macs, K_h x K_w x H x W x C_out
        macs = list(out_shape[1:])
        macs.append(kernel)
        macs = multiply(macs)
        return {'weights': weights, 'macs': macs}

    def _estimate_convolution(self, in_shape, out_shape, params):
        out_info = self._estimate_depthwise_convolution(out_shape, params)
        # input channel size C_in
        in_channels = in_shape[-1]
        out_channels = out_shape[-1]
        out_info['macs'] *= in_channels
        out_info['weights'] = int(
            in_channels * out_info['weights'] + out_channels)
        return out_info

    def _weight_overrider(self, node):
        return self.overriders.get(node, {}).get('weights')

    def estimate_convolution(self, node, in_info, in_shape, out_shape, params):
        out_info = self._estimate_convolution(in_shape, out_shape, params)
        o = self._weight_overrider(node)
        return apply_sparsity(o, in_info, out_info, in_shape, out_shape)

    def estimate_depthwise_convolution(
            self, node, in_info, in_shape, out_shape, params):
        out_info = self._estimate_depthwise_convolution(out_shape, params)
        o = self._weight_overrider(node)
        out_info = apply_sparsity(
            o, in_info, out_info, in_shape, out_shape, depthwise=True)
        return out_info
        # FIXME only works for FBS
        # return passthrough(in_info, out_info)

    def estimate_fully_connected(
            self, node, in_info, in_shape, out_shape, params):
        macs = in_shape[-1] * out_shape[-1]
        out_info = {'macs': macs, 'weights': macs}
        o = self._weight_overrider(node)
        return apply_sparsity(o, in_info, out_info, in_shape, out_shape)

    def estimate_concat(self, node, in_infos, input_shapes, out_shape, params):
        # FIXME only works for FBS
        return {}
        # masks = []
        # for info, shape in zip(in_infos, input_shapes):
        #     hist = info.get('_mask') or np.ones(shape, dtype=bool)
        #     masks.append(hist)
        # mask = []
        # for each in zip(*masks):
        #     mask.append(np.concatenate(each, axis=-1))
        # density, active = mask_density(mask)
        # return {'_mask': mask, 'density': density, 'active': active}

    @staticmethod
    def _estimate_join(masks, reducer):
        mask = mask_join(masks, reducer)
        density, active = mask_density(mask)
        return {'_mask': mask, 'density': density, 'active': active}

    def _estimate_binary_elementwise(self, in_infos, input_shapes, reducer):
        mask_shape = input_shapes[0]
        masks = []
        for i in in_infos:
            hist = i.get('_mask') or np.ones(mask_shape, dtype=bool)
            masks.append(hist)
        return self._estimate_join(masks, reducer)

    def estimate_add(self, node, in_infos, input_shapes, out_shape, params):
        return self._estimate_binary_elementwise(
            in_infos, input_shapes, np.logical_or)

    def estimate_mul(self, node, in_infos, input_shapes, out_shape, params):
        return self._estimate_binary_elementwise(
            in_infos, input_shapes, np.logical_and)

    def _passthrough(self, node, in_info, in_shape, out_shape, params):
        # FIXME only works for FBS
        return passthrough(in_info, {})

    # estimate_dropout = _passthrough
    # estimate_identity = _passthrough
    # estimate_average_pool = _passthrough
    # estimate_max_pool = _passthrough
    # estimate_activation = _passthrough
