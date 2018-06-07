import math
import tensorflow as tf
import numpy as np
import yaml
import pickle

from mayo.log import log
from mayo.session.util import Targets
from mayo.session.train import Train


class SearchBase(Train):
    modes = ['one_shot', 'one_epoch', 'fine_tune']

    def search(self):
        log.debug('Searching starts.')
        try:
            run = self._init_search()
            if run:
                while self._search_iteration():
                    pass
        except KeyboardInterrupt:
            log.info('Stopped.')
            save = self.config.system.checkpoint.get('save', {})
            if save:
                countdown = save.get('countdown', 0)
                if log.countdown('Saving checkpoint', countdown):
                    self.save_checkpoint('latest')

    def _init_search(self):
        self.search_mode = self.config.search.search_mode
        self._reset_stats()
        self.reset_num_epochs()
        if not (self.search_mode in self.modes):
            raise ValueError('search model {} is not one of {}'.format(
                self.search_mode, self.modes))
        if self.search_mode == 'fine_tune':
            # init hyper parameters for fine-tune search
            self.acc_base = self.config.search.parameters.get('acc_base', 0.0)
            self.search_overriders()
            return True
        search_func = getattr(self, 'search_simple')
        search_func()
        return False

    def assign_targets(self, overrider, targets, values):
        for target, value in zip(targets, values):
            setattr(overrider, target, value)
        return

    def parse_range(self, r):
        return [i for i in range(r['from'], r['to'] + r['scale'], r['scale'])]

    def generate_overriders(self, overriders):
        for key, os in overriders.items():
            for o in os:
                yield o

    def search_overriders(self):
        self._init_scales()
        self._reset_stats()
        self.targets.init_targets(
            self, self.config.search.run_status == 'normal')
        self.construct_targets(start=True)
        self.profile_for_one_epoch()
        # init all overriders
        for variable in self.targets.members:
            self.variable_init(variable)
        self.overriders_update()
        return True

    def _search_iteration(self):
        system = self.config.system
        epoch = self.once()
        self._update_stats(self.estimator.get_value('accuracy', 'train'))
        floor_epoch = math.floor(epoch)
        cp_interval = system.checkpoint.get('save.interval', 0)

        # if self.change.every('checkpoint.epoch', floor_epoch, cp_interval):
        if epoch > 0.1:
            self._avg_stats()
            if self.acc_avg >= self.acc_base:
                self.start_acc = None
                return self.forward_policy(floor_epoch)

            iter_max_epoch = self.config.search.get('num_epochs', None)
            if iter_max_epoch is None:
                raise ValueError('num_epochs keyword is missing')

            # current setup exceeds max epoches, retrieve backwards
            early_stop = self.config.search.get('early_stop', False)
            # if self._exceed_max_epoch(epoch, iter_max_epoch, early_stop):
            if self._exceed_max_epoch(epoch, iter_max_epoch, early_stop):
                self.search_cnt += 1
                self.reset_num_epochs()
                self.log_thresholds(self.loss_avg, self.acc_avg)
                self.start_acc = None
                return self.backward_policy()
        return True

    def _exceed_max_epoch(self, epoch, max_epoch, early_stop=None):
        if early_stop:
            baseline = (self.acc_base - self.start_acc) / float(max_epoch)
            acc_grad = (self.acc_avg - self.start_acc) / float(epoch)
            # increasing grad but slower than expected
            if acc_grad > 0 and baseline > 0 \
                    and acc_grad < (baseline * early_stop):
                log.info('Early stop activated !')
                return True
        return epoch >= max_epoch and epoch > 0

    def _fetch_as_overriders(self, info):
        self.targets = Targets(info)
        for key, o in self.task.nets[0].overriders.items():
            if isinstance(o, list):
                for each_o in o:
                    self.targets.add_overrider(each_o)
                continue
            self.targets.add_overrider(o)

    def _fetch_as_variables(self, info):
        self.targets = Targets(info)
        targeting_vars = []
        associated_vars = []
        for name in info.targets:
            for item in self.global_variables():
                if name in item.name:
                    targeting_vars.append(item)
        for name in info.associated:
            for item in self.global_variables():
                if name in item.name:
                    associated_vars.append(item)
        for zipped in zip(targeting_vars, associated_vars):
            self.targets.add_target(*zipped)

    def _node_logging(self, write_to_files):
        if not hasattr(self, 'writing_cnt'):
            self.writing_cnt = 0
        else:
            self.writing_cnt += 1
        filename = 'node_log/log' + str(self.writing_cnt) + '.pkl'
        nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
        ops = [op.name for op in tf.get_default_graph().get_operations()]
        with open(filename, 'wb') as f:
            pickle.dump([nodes, ops], f)

    def _init_scales(self):
        self.log = {}
        self.search_cnt = 0
        self.target_layer = None
        self.loss_avg = None
        self.best_ckpt = None
        info = self.config.search.parameters
        mode = info.get('overriders', True)
        # define initial scale
        if mode:
            self._fetch_as_overriders(info)
        else:
            self._fetch_as_variables(info)

    def once(self):
        train_op = self._train_op if self._run_train_ops else []
        if self.config.search.get('eval_only', False):
            # do not run training operations when `search.eval_only` is set
            train_op = train_op['imgs_seen']
        tasks = [train_op, self._mean_loss, self.num_epochs]
        noop, loss, num_epochs = self.run(tasks, batch=True)
        self.running_loss = loss
        if math.isnan(loss):
            raise ValueError('Model diverged with a nan-valued loss.')
        return num_epochs

    def backward_policy(self):
        raise NotImplementedError(
            'Method of backward policy is not implemented.')

    def forward_policy(self, floor_epoch):
        raise NotImplementedError(
            'Method of forward policy is not implemented.')

    def _fetch_scale(self):
        for item in self.targets.members:
            if item.name == self.target_layer.name:
                return item.scale

    def construct_targets(self, start=False):
        '''
        1. profile the associated vars and determine a priority list
        2. produce a cont list to determine which targeting vars continues on
        searching
        '''
        if start:
            self.best_ckpt = self.config.system.checkpoint.load
            # if yaml exists, load it and compute self.cont
            if self.config.search.get('cont_list'):
                doct_cont = self.config.search.cont_list
                self.targets.cont_list(doct_cont)
            # yaml does not exist, initialize to true by default
            else:
                self.targets.cont_list()
        # pick next target
        self.target_layer = self.targets.pick_layer(self, start)

    def _prepare_yaml(self, write):
        name = self.config.model.name
        self.stream = open(
            'trainers/{}_search_base.yaml'.format(name), 'w')
        self.dump_data = {
            'search': {
                'acc_base': float(self.acc_base)}}
        if write:
            self.stream.write(yaml.dump(
                self.dump_data,
                default_flow_style=False))
        return

    def profile_for_one_epoch(self):
        epoch = 0
        self._reset_stats()
        self.reset_num_epochs()

        self.acc_base = self.config.search.get('acc_base')
        # if acc is hand coded in yaml
        if self.acc_base:
            log.info(
                'loaded profiled acc baseline, acc is {}'
                .format(self.acc_base))
            self._prepare_yaml(False)
            return

        tolerance = self.config.search.tolerance
        log.info('Profiling baseline')
        tasks = [self._mean_loss, self.num_epochs]
        # while epoch < 1.0:
        while epoch < 0.1:
            loss, epoch = self.run(tasks, batch=True)
            self.loss_total += loss
            self.acc_total += self.estimator.get_value('accuracy')
            self.step += 1

        self.acc_base = self.acc_total / float(self.step) * (1 - tolerance)

        log.info('Profiled accuracy is {}'.format(self.acc_base))
        self._reset_stats()
        self.reset_num_epochs()
        self._prepare_yaml(True)

    def _avg_stats(self):
        self.acc_avg = self.acc_total / float(self.step)

    def _update_stats(self, acc):
        if not hasattr(self, 'start_acc') or self.start_acc is None:
            self.start_acc = acc
        self.step += 1
        self.acc_total += acc

    def _reset_stats(self):
        self.step = 0
        self.acc_total = 0
        self.loss_total = 0

    def variables_refresh(self):
        raise NotImplementedError(
            'Method to refresh overriders is not implemented.')

    def variable_init(self, target):
        variables = target.tv
        thresholds = target.thresholds
        for v, t in zip(variables, thresholds):
            self.assign(v, t)

    def variable_refresh(self, target):
        variables = target.tv
        scale = target.scale
        target.thresholds = [tmp + scale for tmp in target.thresholds]
        for v, t in zip(variables, target.thresholds):
            self.assign(v, t)

    def log_thresholds(self, loss, acc):
        _, _, prev_loss = self.log.get(self.target_layer, [None, None, None])
        for tv in self.targets.members:
            value = tv.thresholds
            if prev_loss is None:
                self.log[tv.name] = (value, loss, acc)
            else:
                if acc > self.acc_base:
                    self.log[self.target_layer] = (value, loss, acc)

    def np_quantize_loss(self, before, after):
        return np.sum(np.abs(after - before))
