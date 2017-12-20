import tensorflow as tf
import operator
import numpy as np

from mayo.log import log
from mayo.override.base import OverriderBase, ChainOverrider
from mayo.override.quantize import Recentralizer


class Targets(object):
    def __init__(self, info):
        self.info = info
        self.members = []

    def _check_type(self, overrider):
        # pin type, !! does not check chained overrider!!
        o_type = self.info.type
        # check recentralizer
        if hasattr(overrider, 'quantizer'):
            o_name = overrider.quantizer.__class__.__name__
        else:
            o_name = overrider.__class__.__name__
        if o_name in o_type:
            return True
        return False

    def _add_chain_overrider(self, element):
        for chained in element:
            if self._check_type(chained):
                self._instantiate_target(chained)

    def _instantiate_target(self, element):
        target = self.info.target
        if isinstance(element, Recentralizer):
            tv = getattr(element.mean_quantizer, target)
            self.members.append(Target(tv, element))
            element = element.quantizer
        tv = getattr(element, target)
        self.members.append(Target(tv, element))

    def add_overrider(self, element):
        if isinstance(element, ChainOverrider):
            self._add_chain_overrider(element)
        else:
            self._instantiate_target(element)

    def add_target(self, tv, element):
        self.members.append(Target(tv, element))

    def show_targets(self):
        return [item.tv for item in self.members]

    def show_associates(self):
        return [item.av for item in self.members]

    def target_iterator(self):
        for item in self.members:
            yield (item.tv, item.av)

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

    def sort_targets(self, session):
        self.members = sorted(self.members, key=lambda x: x.metric(session))


class Target(object):
    def __init__(self, target_var, associate_var):
        self.tv = target_var
        self.av = associate_var

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


class Info(object):
    def __init__(self, meta_info, session, targeting_vars, run_status):
        self.scales = {}
        self.min_scales = {}
        self.scale_update_factors = {}
        self.start_ths = {}
        self.ths = {}
        self.max_ths = {}

        # now only supports retraining on one overrider
        self.meta = meta_info

        for target in targeting_vars:
            name = target.name
            self.scales[name] = meta_info.range['scale']
            self.min_scales[name] = meta_info.range['min_scale']
            self.scale_update_factors[name] = \
                meta_info.range['scale_update_factor']

            if run_status == 'continue':
                th = session.run(target)
                self.ths[name] = th
                self.start_ths[name] = th
                log.info('{} is continuing on {}.'.format(name, th))
            else:
                self.ths[name] = meta_info.range['from']
                self.start_ths[name] = meta_info.range['from']
            self.max_ths[name] = meta_info.range['to']
        if self.scales == {}:
            raise ValueError(
                '{} is not found in overrider definitions, '
                'but has been specified as a target.'.format(meta_info.type))

    def get(self, variable, info_type):
        name = variable.name
        if info_type == 'end_threshold':
            return self.max_ths[name]
        if info_type == 'end_scale':
            return self.min_scales[name]
        if info_type == 'threshold':
            return self.ths[name]
        if info_type == 'start_threshold':
            return self.start_ths[name]
        if info_type == 'scale':
            return self.scales[name]
        if info_type == 'scale_factor':
            return self.scale_update_factors[name]
        raise ValueError('{} is not a collected info key.'.format(info_type))

    def set(self, variable, info_type, value):
        name = variable.name
        if info_type == 'threshold':
            self.ths[name] = value
        elif info_type == 'scale':
            self.scales[name] = value
        else:
            raise ValueError(
                '{} is not a collected info key.'.format(info_type))
