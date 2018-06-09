import yaml
import numpy as np
import tensorflow as tf

from mayo.session.search.base import SearchBase
from mayo.session.profile import Profile
from mayo.override.base import OverriderBase
from mayo.log import log
from mayo.util import Table


class LayerwiseSearch(SearchBase, Profile):
    def forward_policy(self, floor_epoch):
        log.debug('Targeting on {}...'.format(self.target_layer.name))
        log.debug('Log: {}'.format(self.log))
        self.best_ckpt = 'search-' + str(self.search_cnt) + '-' \
            + str(floor_epoch)
        self.save_checkpoint(self.best_ckpt)
        self._cp_epoch = floor_epoch
        self.search_cnt += 1
        self.log_thresholds(self.loss_avg, self.acc_avg)
        self.construct_targets()
        self.variables_refresh()
        self.reset_num_epochs()
        for item in self.targets.members:
            if item.name == self.target_layer.name:
                for tv in item.tv:
                    threshold = item.thresholds[0]
                    break
        log.info(
            'update threshold to {}, working on {}'
            .format(threshold, self.target_layer.name))
        return True

    def backward_policy(self):
        finished = self.target_layer.name in self.targets.cont
        if self.targets.priority_list == [] and finished:
            log.info(
                'Done, model stored at {}.'.format(self.best_ckpt))
            for item in self.targets.members:
                names = [tv.name for tv in item.tv]
                thresholds = item.thresholds
                scale = item.scale
                log.info(
                    'Layer name: {}, threshold: {}, scale: {}.'
                    .format(names, thresholds, scale))
            return False
        else:
            # trace back
            self.load_checkpoint(self.best_ckpt)
            # current layer is done
            for item in self.targets.members:
                if item.name == self.target_layer.name:
                    recorded = item
                    break
            # assuming only one threshold is targeted per layer
            min_scale = recorded.min_scale
            scale = recorded.scale
            end_thresholds = recorded.end_thresholds
            thresholds = recorded.thresholds

            if scale > 0:
                scale_check = self._fetch_scale() > min_scale
                threshold_check = [e > t for t, e in zip(
                    thresholds, end_thresholds)]
                threshold_check = any(threshold_check)
            elif scale == 0:
                scale_check = False
                threshold_check = False
            else:
                scale_check = self._fetch_scale() < min_scale
                threshold_check = [e < t for t, e in zip(
                    thresholds, end_thresholds)]
                threshold_check = any(threshold_check)
            run = scale_check and threshold_check
            if run:
                # overriders are refreshed inside decrease scale
                self._decrease_scale()
                self.reset_num_epochs()
                log.info(
                    'Decreasing scale to {}, working on {}...'
                    .format(self._fetch_scale(), self.target_layer.name))
            else:
                # threshold roll back
                for item in self.targets.members:
                    if item.name == self.target_layer.name:
                        item.thresholds = [threshold - item.scale for
                                           threshold in item.thresholds]
                # give up on current target
                self.targets.cont = [tmp for tmp in self.targets.cont if
                                     tmp != self.target_layer.name]
                # fetch a new layer to search
                self.construct_targets()
                self.variables_refresh()
                self.reset_num_epochs()
                if not threshold_check:
                    log.info('threshold meets its minimum')
                if not scale_check:
                    log.info('scale meets its minimum')
                log.info('switching layer, working on {}'.format(
                    self.target_layer.name))
                log.info('priority_list: {}'.format(
                    self.targets.priority_list))
                # refresh the yaml
                self.dump_data['search']['cont_list'] = self.targets.cont
                yaml.dump(
                    self.dump_data, self.stream, default_flow_style=False)
            return True

    def variables_refresh(self):
        update_flag = False
        for item in self.targets.members:
            if item.name == self.target_layer.name:
                self.variable_refresh(item)
                if isinstance(item.av, OverriderBase):
                    item.av.should_update = True
                    update_flag = True
        if update_flag:
            self.overriders_update()

    def _decrease_scale(self):
        # decrease scale factor, for quantizer, this factor might be 1
        for item in self.targets.members:
            if item.name == self.target_layer.name:
                # threshold roll back
                scale = item.scale
                start_th = item.start_threshold
                check_start = all([th == start_th for th in item.thresholds])
                if check_start:
                    raise ValueError(
                        'Threshold failed at starting point, consider '
                        'changing your starting point.')
                else:
                    item.thresholds = [th - scale for th in item.thresholds]
                factor = item.update_factor
                # update scale
                item.scale = self._new_scale(scale, item.thresholds, factor)
                self.variable_refresh(item)
        self.overriders_update()

    @staticmethod
    def _new_scale(old_scale, thresholds, factor):
        check_threshold = all([isinstance(th, int) for th in thresholds])
        if isinstance(old_scale, int) and check_threshold:
            return int(old_scale * factor)
        return old_scale * factor

    def search_simple(self, search_mode):
        print('Search progressing ...')
        config = self.config.search
        overriders = self.task.nets[0].overriders
        name_to_rules = config.parameters.overriders
        # training = config.pop('training', False)
        export_ckpt = config.pop('export_ckpt', False)
        macs = self.task.nets[0].estimate()
        priority_ranks = [(key, macs[key]) for key, o in overriders.items()]
        priority_ranks = sorted(
            priority_ranks, key=lambda x:x[1]['macs'], reverse=True)
        target_values = {}
        if search_mode == 'one_shot':
            for key, _ in priority_ranks:
                for o in overriders[key]:
                    target_values[o] = {}
                    o.update()
                    for keyword, rules in name_to_rules.items():
                        if keyword == type(o).__name__:
                            for target in rules.targets:
                                target_values[o][target] = self.run(
                                    getattr(o, target))
        else:
            target_values = self.profiled_search(
                config.training, overriders, name_to_rules)
        self.present(overriders, target_values, export_ckpt)
        return False

    def profiled_search(self, training, overriders, rules):
        # decide to train or not
        search_params = self.config.search.parameters
        self._run_train_ops = training

        self.config.system.max_epochs = search_params.profile.start
        # empty run to speed to warm up
        for o, key in self.generate_overriders(overriders, prod_key=True):
            o.enable = False
            o.width = 8
        self.profile()
        # lets profile the values
        self.register_values(
            overriders, samples=search_params.samples,
            rules=rules)
        self.config.system.max_epochs = search_params.profile.end
        self.profile(reset=False)
        meta_params = {}
        targets = {}
        for o, key in self.generate_overriders(overriders, prod_key=True):
            # construct after, overrde again
            params = {}
            avg = self.estimator.get_value('avg_' + o.name, node=key)
            max_val = self.estimator.get_value('max_' + o.name, node=key)
            params['max'] = max_val[0]
            params['avg'] = avg
            params['samples'] = self.estimator.get_value(o.name, node=key)
            meta_params[o.name] = params
            target = o.search(params)
            targets[o] = target
        return targets

    def register_values(
            self, overriders, reg_avg=True, reg_max=True, samples=10,
            rules=None):
        for o, key in self.generate_overriders(overriders, prod_key=True):
            name = type(o).__name__
            if reg_avg:
                self.estimator.register(
                    o.before, 'avg_' + o.name, node=key,
                    history='running_mean')
            if reg_max:
                percentile = rules[name].percentile
                percentile = tf.contrib.distributions.percentile(
                    tf.abs(o.before), percentile)
                self.estimator.register(
                    percentile, 'max_' + o.name, node=key,
                    history='running_mean')
            self.estimator.register(
                o.before, o.name, node=key, history=samples)
        return

    def present(self, overriders, target_values, export_ckpt):
        table = Table(['variable', 'suggested value'])
        for o in self.generate_overriders(overriders):
            name = o.name
            if len(name) > 4:
                name = o.name.split('/')
                name = '/'.join(name[-4:])
            table.add_row((
                name, target_values[o]))
        print(table.format())
        if export_ckpt:
            model_name = self.config.model.name
            model_name += '_profile_' + self.config.search.search_mode
            self.save_checkpoint(model_name)
        return
