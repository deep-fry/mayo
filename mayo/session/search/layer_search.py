import yaml
import numpy as np
import tensorflow as tf
from itertools import product

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
        session = self.task.session
        overriders = self.task.nets[0].overriders
        targets = config.parameters.target
        ranges = config.parameters.range
        num_epochs = config.pop('num_epochs', 'one_shot')
        training = config.pop('training', False)
        export_ckpt = config.pop('export_ckpt', False)
        link_width = self.config.search.parameters.pop('link_width', None)
        if len(ranges) == 1:
            ranges = len(targets) * ranges
        ranges = [self.parse_range(r) for r in ranges]
        q_losses = {}
        items = {}
        # lets decide priority first
        macs = self.task.nets[0].estimate()
        priority_ranks = [(key, macs[key]) for key, o in overriders.items()]
        priority_ranks = sorted(
            priority_ranks, key=lambda x:x[1]['macs'], reverse=True)
        for key, _ in priority_ranks:
            for o in overriders[key]:
                q_loss = 0
                q_losses[o.name] = []
                items[o.name] = []
                for item in product(*ranges):
                    if link_width and item[link_width[0]] > item[link_width[1]]:
                        continue
                    self.assign_targets(o, targets, item)
                    if num_epochs == 'one_shot' or search_mode == 'one_shot':
                        before, after = session.run([o.before, o.after])
                        q_loss = self.np_quantize_loss(before, after)
                    else:
                        q_loss = self.profiled_search(
                            training, num_epochs, o, targets, item)
                    q_losses[o.name].append(q_loss)
                    items[o.name].append(item)
        self.present(overriders, items, q_losses, targets, export_ckpt)
        return False

    def profiled_search(
            self, training, num_epochs, overrider, targets, item):
        overriders = self.task.nets[0].overriders
        self.flush_quantize_loss(overriders)
        # decide to train or not
        self._run_train_ops = training
        if isinstance(num_epochs, (int, float)):
            self.config.system.max_epochs = num_epochs
        self.assign_targets(overrider, targets, item)
        # registered quantization loss
        self.profile(overriders)
        return self.estimator.get_value(overrider.name)[0]

    def present(self, overriders, items, losses, targets, export_ckpt):
        table = Table(['variable', 'suggested value', 'target', 'loss'])
        filter_variables = []
        for key, os in overriders.items():
            for o in os:
                for target in targets:
                    sel_loss = np.min(np.array(losses[o.name]))
                    sel_arg = np.argmin(np.array(losses[o.name]))
                    name = o.name
                    if len(name) > 4:
                        name = o.name.split('/')
                        name = '/'.join(name[-4:])
                    if isinstance(getattr(o, target), tf.Variable):
                        filter_variables.append(getattr(o, target))
                    table.add_row((
                        name, items[o.name][sel_arg], target, sel_loss))
        if len(filter_variables) and export_ckpt:
            self.save_variables(filter_variables)
        print(table.format())
        return

    def save_variables(self, variables):
        log.info('Saving suggested targeting values to a checkpoint')
        saver = tf.train.Saver(variables)
        saver.save(self.tf_session, './suggestion', write_meta_graph=False)
