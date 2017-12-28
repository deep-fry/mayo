import tensorflow as tf
import operator
import numpy as np
import re

from mayo.log import log
from mayo.util import import_from_string
from mayo.override.base import OverriderBase, ChainOverrider
from mayo.override.quantize import Recentralizer


class Targets(object):
    def __init__(self, info):
        self.info = info
        # simply keep all members
        self.members = []
        # priority list, a subset of members and is sorted with respect to avs
        self.priority_list = []

    def _check_type(self, overrider):
        o_type = self.info.type
        targets = [import_from_string(item) for item in o_type]
        # check recentralizer
        if isinstance(overrider, Recentralizer):
            o = overrider.quantizer
        else:
            o = overrider
        for target in targets:
            if isinstance(o, target):
                return True
        return False

    def _add_chain_overrider(self, element):
        for chained in element:
            if self._check_type(chained):
                self._instantiate_target(chained, name=element.name)

    @staticmethod
    def _is_bias(variable):
        return 'bias' in variable.name

    def _instantiate_target(self, element, name=None):
        target = self.info.target
        tvs = []
        mean_quantizer = None
        quantizer = None
        bias_quantizer = None
        if isinstance(element, Recentralizer):
            if self._is_bias(element):
                bias_quantizer = element
                tvs.append(getattr(element, target))
            else:
                mean_quantizer = element.mean_quantizer
                quantizer = element.quantizer
                tvs.append(getattr(element.mean_quantizer, target))
                tvs.append(getattr(element.quantizer, target))
        else:
            if self._is_bias(element):
                bias_quantizer = element
            tvs.append(getattr(element, target))

        if name is None:
            name = self._check_parent_layer(element.name)
        else:
            name = self._check_parent_layer(name)
        for existing_target in self.members:
            # current target has associates in self.members
            if name == existing_target.name:
                existing_target.add_var(tvs)
                if self._is_bias(element):
                    existing_target.bias_quantizer = bias_quantizer
                else:
                    existing_target.mean_quantizer = mean_quantizer
                    existing_target.quantizer = quantizer
                return
        # current target does not exsit, creat it
        self.members.append(Target(
            tvs, element, name, quantizer, mean_quantizer, bias_quantizer))

    def add_overrider(self, element):
        if isinstance(element, ChainOverrider):
            self._add_chain_overrider(element)
        else:
            self._instantiate_target(element)

    def add_member(self, tv, element):
        self.members.append(Target(tv, element))

    def show_targets(self):
        return [item.tv for item in self.members]

    def show_associates(self):
        return [item.av for item in self.members]

    def update_targets(self, action, mode='tv'):
        if mode == 'all':
            for item in self.members:
                action(item.tv)
                action(item.av)
        else:
            for item in self.members:
                action(getattr(item, mode))

    def change_property(self, attr, value, mode='tv'):
        '''
        args:
            p_name: name of the targeting attribute
            value: the new value
        '''
        for item in self.members:
            setattr(getattr(item, mode), attr, value)

    def sort_layers(self, session):
        # filter out elements that are not in self.cont
        priority_list = []
        for item in self.members:
            if item.name in self.cont:
                priority_list.append(item)
        self.priority_list = sorted(
            priority_list, key=lambda x: x.metric(session))

    def cont_list(self, load_doct=None):
        self.cont = []
        if load_doct is not None:
            for variable in self.members:
                if load_doct.get(variable.name):
                    self.cont.append(variable.name)
        else:
            self.cont = [variable.name for variable in self.members]
        log.info('continue on layers: {}'.format(self.cont))

    def pick_layer(self, session, start=False):
        if self.priority_list == [] and not start:
            log.info('priority list is empty!!')
            return None
        else:
            self.sort_layers(session)
            if start:
                log.info('first time picking targets: {}'.format(
                    self.priority_list))
            return self.priority_list.pop()

    def _check_parent_layer(self, overrider_name):
        datepat = re.compile(r'(.*?)/(.*?)/(.*?)')
        m = datepat.match(overrider_name)
        net_name, layer_name, _ = m.groups()
        return layer_name

    def init_targets(self, session, start):
        for item in self.members:
            item.init_info(self.info, session, start)


class Target(object):
    def __init__(
            self, target_vars, associate_var, name, quantizer,
            mean_quantizer, bias_quantizer):
        self.tv = target_vars
        self.av = associate_var
        self.name = name
        # extra collections
        self.mean_quantizer = mean_quantizer
        self.quantizer = quantizer
        self.bias_quantizer = bias_quantizer

    def init_info(self, info, session, start):
        self.scale = info.range['scale']
        self.min_scale = info.range['min_scale']
        self.update_factor = info.range['scale_update_factor']
        self.start_threshold = info.range['from']
        if start:
            self.thresholds = [info.range['from']] * len(self.tv)
        else:
            self.thresholds = [session.run(v) for v in self.tv]
        self.end_thresholds = [info.range['to']] * len(self.tv)

    def add_var(self, variable):
        self.tv.extend(variable)

    def metric(self, session):
        return self._metric_clac(self.av, session)

    def _metric_clac(self, variable, session):
        if isinstance(variable, (tf.Variable, tf.Tensor)):
            return session.run(variable).size
        if isinstance(variable, ChainOverrider):
            chained_vars = [self._metric_calc(v) for v in variable]
            metric_value = reduce(operator.mul, chained_vars)
        if isinstance(variable, OverriderBase):
            if hasattr(variable, 'width'):
                return session.run(variable.width)
            if hasattr(variable, 'mask'):
                density = np.count_nonzero(session.run(variable.mask)) / \
                    float(session.run(variable.after).size)
                return density
            else:
                metric_value = 1
        # normal tensor value should have been returned
        return metric_value * session.run(variable.after).size

    @staticmethod
    def has_mean(variable):
        if 'mean' in variable.name:
            return True

    def pick_mean(self, variable):
        for item in self.tv:
            if self.has_mean(item):
                return item
#
# class Info(object):
#     def __init__(self, meta_info, session, targeting_vars, run_status):
#         self.scales = {}
#         self.min_scales = {}
#         self.scale_update_factors = {}
#         self.start_ths = {}
#         self.ths = {}
#         self.max_ths = {}
#
#         # now only supports retraining on one overrider
#         self.meta = meta_info
#
#         for target in targeting_vars:
#             name = target.name
#             self.scales[name] = meta_info.range['scale']
#             self.min_scales[name] = meta_info.range['min_scale']
#             self.scale_update_factors[name] = \
#                 meta_info.range['scale_update_factor']
#
#             if run_status == 'continue':
#                 th = session.run(target)
#                 self.ths[name] = th
#                 self.start_ths[name] = th
#                 log.info('{} is continuing on {}.'.format(name, th))
#             else:
#                 self.ths[name] = meta_info.range['from']
#                 self.start_ths[name] = meta_info.range['from']
#             self.max_ths[name] = meta_info.range['to']
#         if self.scales == {}:
#             raise ValueError(
#                 '{} is not found in overrider definitions, '
#                 'but has been specified as a target.'.format(meta_info.type))
#
#     def get(self, variable, info_type):
#         name = variable.name
#         if info_type == 'end_threshold':
#             return self.max_ths[name]
#         if info_type == 'end_scale':
#             return self.min_scales[name]
#         if info_type == 'threshold':
#             return self.ths[name]
#         if info_type == 'start_threshold':
#             return self.start_ths[name]
#         if info_type == 'scale':
#             return self.scales[name]
#         if info_type == 'scale_factor':
#             return self.scale_update_factors[name]
#         raise ValueError('{} is not a collected info key.'.format(info_type))
#
#     def set(self, variable, info_type, value):
#         name = variable.name
#         if info_type == 'threshold':
#             self.ths[name] = value
#         elif info_type == 'scale':
#             self.scales[name] = value
#         else:
#             raise ValueError(
#                 '{} is not a collected info key.'.format(info_type))
