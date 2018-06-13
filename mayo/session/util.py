import re
import operator

import numpy as np
import tensorflow as tf

from mayo.log import log
from mayo.util import import_from_string
from mayo.override.base import OverriderBase, ChainOverrider
from mayo.override.quantize import Recentralizer


class Targets(object):
    def __init__(self, info, session):
        self.info = info
        # targets to associated map
        self.associates = {}
        self.ranges = {}
        self.priority_list = []
        self.session = session
        overriders_targets = info.pop('overriders', None)
        self.add_targets(overriders_targets, session, overrider=True)
        variables_targets = info.pop('variables', None)
        self.add_targets(variables_targets, session, overrider=False)

    def add_target(self, target_meta_info, session, overrider=True):
        target_name = target_meta_info.pop('targets', None)
        if target_name is None:
            log.error('Specify a target for overrider.')
        if len(target_name) != 1:
            log.error(
                'Fine tuning only supports single target but has {}'.format(
                    target_name))
        else:
            target_name = target_name[0]
        if overrider:
            for key, meta_info in target_meta_info:
                for okey, overrider in self.session.overriders():
                    for o in overrider:
                        if key in type(o).__name__:
                            target = getattr(o, target_name)
                            self.associates[target] = o.after
        else:
            for key, meta_info in target_meta_info:
                target = associate = None
                for variable in session.global_variables():
                    if key == variable.name:
                        associate = variable
                    if target_name == variable.name:
                        target = variable
                if target is None:
                    log.error('target {} does not exsits in global variables.')
                if associate is None:
                    log.error(
                        'associate {} does not exsits in global variables.')
                self.associate[target] = associate
        for target in self.associates.keys():
            self.ranges[target] = {}




#
#     def _check_type(self, overrider):
#         o_type = self.info.type
#         targets = [import_from_string(item) for item in o_type]
#         # check recentralizer
#         if isinstance(overrider, Recentralizer):
#             o = overrider.quantizer
#         else:
#             o = overrider
#         for target in targets:
#             if isinstance(o, target):
#                 return True
#         return False
#
#     def _add_chain_overrider(self, element):
#         for chained in element:
#             if self._check_type(chained):
#                 self._instantiate_target(chained, name=element.name)
#
#     @staticmethod
#     def _is_bias(variable):
#         return 'bias' in variable.name
#
#     def _instantiate_target(self, element, name=None):
#         target = self.info.overriders.get(type(element).__name__, None)
#         target = target.get('targets')
#         if target is None:
#             log.error('Specify a target for overrider {}'.format(element))
#         if len(target) != 1:
#             log.error(
#                 'Fine tuning only supports single target but has {}'.format(
#                     target))
#         else:
#             target = target[0]
#         tvs = []
#         mean_quantizer = None
#         quantizer = None
#         bias_quantizer = None
#         if isinstance(element, Recentralizer):
#             if self._is_bias(element):
#                 bias_quantizer = element
#                 tvs.append(getattr(element, target))
#             else:
#                 mean_quantizer = element.mean_quantizer
#                 quantizer = element.quantizer
#                 tvs.append(getattr(element.mean_quantizer, target))
#                 tvs.append(getattr(element.quantizer, target))
#         else:
#             if self._is_bias(element):
#                 bias_quantizer = element
#             tvs.append(getattr(element, target))
#
#         if name is None:
#             name = self._check_parent_layer(element.name)
#         else:
#             name = self._check_parent_layer(name)
#         for existing_target in self.members:
#             # current target has associates in self.members
#             if name == existing_target.name:
#                 existing_target.add_var(tvs)
#                 if self._is_bias(element):
#                     existing_target.bias_quantizer = bias_quantizer
#                 else:
#                     existing_target.mean_quantizer = mean_quantizer
#                     existing_target.quantizer = quantizer
#                 return
#         # current target does not exsit, creat it
#         self.members.append(Target(
#             tvs, element, name, quantizer, mean_quantizer, bias_quantizer))
#
#     def add_overrider(self, element):
#         if isinstance(element, ChainOverrider):
#             self._add_chain_overrider(element)
#         else:
#             self._instantiate_target(element)
#
#     def add_member(self, tv, element):
#         self.members.append(Target(tv, element))
#
#     def show_targets(self):
#         return [item.tv for item in self.members]
#
#     def show_associates(self):
#         return [item.av for item in self.members]
#
#     def update_targets(self, action, mode='tv'):
#         if mode == 'all':
#             for item in self.members:
#                 action(item.tv)
#                 action(item.av)
#         else:
#             for item in self.members:
#                 action(getattr(item, mode))
#
#     def change_property(self, attr, value, mode='tv'):
#         '''
#         args:
#             p_name: name of the targeting attribute
#             value: the new value
#         '''
#         for item in self.members:
#             setattr(getattr(item, mode), attr, value)
#
#     def sort_layers(self, session):
#         # filter out elements that are not in self.cont
#         priority_list = []
#         for item in self.members:
#             if item.name in self.cont:
#                 priority_list.append(item)
#         self.priority_list = sorted(
#             priority_list, key=lambda x: x.metric(session))
#
#     def cont_list(self, load_doct=None):
#         self.cont = []
#         if load_doct is not None:
#             for variable in self.members:
#                 if load_doct.get(variable.name):
#                     self.cont.append(variable.name)
#         else:
#             self.cont = [variable.name for variable in self.members]
#         log.info('continue on layers: {}'.format(self.cont))
#
#     def pick_layer(self, session, start=False):
#         if self.priority_list == [] and not start:
#             log.info('priority list is empty!!')
#             return None
#         else:
#             self.sort_layers(session)
#             if start:
#                 log.info('First time picking targets: {}'.format(
#                     self.priority_list))
#             return self.priority_list.pop()
#
#     def _check_parent_layer(self, overrider_name):
#         datepat = re.compile(r'(.*?)/(.*?)/(.*?)')
#         m = datepat.match(overrider_name)
#         net_name, layer_name, _ = m.groups()
#         return layer_name
#
#     def init_targets(self, session, start):
#         for item in self.members:
#             item.init_info(self.info, session, start)
#
#
# class Target(object):
#     def __init__(
#             self, target_vars, associate_var, name, quantizer,
#             mean_quantizer, bias_quantizer):
#         self.tv = target_vars
#         self.av = associate_var
#         self.name = name
#         # extra collections
#         self.mean_quantizer = mean_quantizer
#         self.quantizer = quantizer
#         self.bias_quantizer = bias_quantizer
#
#     def init_info(self, info, session, start):
#         import pdb; pdb.set_trace()
#         inforange = info.range[0]
#         self.scale = inforange['scale']
#         self.min_scale = inforange['min_scale']
#         self.update_factor = inforange['scale_update_factor']
#         self.start_threshold = inforange['from']
#         if start:
#             self.thresholds = [inforange['from']] * len(self.tv)
#         else:
#             self.thresholds = [session.run(v) for v in self.tv]
#         self.end_thresholds = [inforange['to']] * len(self.tv)
#
#     def add_var(self, variable):
#         self.tv.extend(variable)
#
#     def metric(self, session):
#         return self._metric_clac(self.av, session)
#
#     def _metric_clac(self, variable, session):
#         if isinstance(variable, (tf.Variable, tf.Tensor)):
#             return session.run(variable).size
#         if isinstance(variable, ChainOverrider):
#             chained_vars = [self._metric_calc(v) for v in variable]
#             metric_value = reduce(operator.mul, chained_vars)
#         if isinstance(variable, OverriderBase):
#             metric_value = 1
#             if hasattr(variable, 'width'):
#                 metric_value *= session.run(variable.width)
#             if hasattr(variable, 'mask'):
#                 density = np.count_nonzero(session.run(variable.mask)) / \
#                     float(session.run(variable.after).size)
#                 metric_value *= density
#         # normal tensor value should have been returned
#         return metric_value * session.run(variable.after).size
#
#     @staticmethod
#     def has_mean(variable):
#         if 'mean' in variable.name:
#             return True
#
#     def pick_mean(self, variable):
#         for item in self.tv:
#             if self.has_mean(item):
#                 return item
