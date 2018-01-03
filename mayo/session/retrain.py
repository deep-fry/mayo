import math
import numpy as np
import tensorflow as tf
import yaml
import pickle
import operator

from mayo.log import log
from mayo.session.util import Targets
from mayo.session.train import Train
from mayo.override.base import OverriderBase, ChainOverrider
from mayo.override.quantize import Recentralizer, FloatingPointQuantizer
from mayo.override.quantize import ShiftQuantizer


class RetrainBase(Train):
    def retrain(self):
        log.debug('Retraining starts.')
        try:
            self._init_retrain()
            while self._retrain_iteration():
                pass
        except KeyboardInterrupt:
            log.info('Stopped.')
            save = self.config.system.checkpoint.get('save', {})
            if save:
                countdown = save.get('countdown', 0)
                if log.countdown('Saving checkpoint', countdown):
                    self.save_checkpoint('latest')

    def _init_retrain(self):
        self.retrain_mode = self.config.retrain.retrain_mode
        self._init_scales()
        self._reset_stats()
        self.targets.init_targets(
            self, self.config.retrain.run_status == 'normal')

        self.profile_associated_vars(start=True)
        self.profile_for_one_epoch()
        # init all overriders
        for variable in self.targets.members:
            self.variable_init(variable)
        self.overriders_update()

    def _retrain_iteration(self):
        system = self.config.system
        loss, acc, epoch = self.once()
        self._update_stats(loss, acc)
        summary_delta = self.change.delta('summary.epoch', epoch)

        if not hasattr(self, 'start_acc') or self.start_acc is None:
            self.start_acc = acc

        if system.summary.save and summary_delta >= 0.1:
            self._save_summary(epoch)
        floor_epoch = math.floor(epoch)
        cp_interval = system.checkpoint.get('save.interval', 0)

        # if epoch > 0.1:
        if self.change.every('checkpoint.epoch', floor_epoch, cp_interval):
            self._avg_stats()
            if self.acc_avg >= self.acc_base:
                self.start_acc = None
                return self.forward_policy(floor_epoch)

            iter_max_epoch = self.config.retrain.iter_max_epoch

            # current setup exceeds max epoches, retrieve backwards
            # if epoch > 0.11:
            # if epoch >= iter_max_epoch and epoch > 0:
            early_stop = self.config.retrain.get('early_stop', False)
            if self._exceed_max_epoch(epoch, iter_max_epoch, early_stop):
                self.retrain_cnt += 1
                self.reset_num_epochs()
                self.log_thresholds(self.loss_avg, self.acc_avg)
                self.start_acc = None
                return self.backward_policy()
                # return self.backward_policy()
        return True

    def _exceed_max_epoch(self, epoch, max_epoch, early_stop=False):
        if early_stop:
            baseline = (self.acc_base - self.start_acc) / float(max_epoch)
            acc_grad = (self.acc_avg - self.start_acc) / float(epoch)
            if acc_grad < (baseline * 0.7):
                log.info('early stop activated')
                return True
        return epoch >= max_epoch and epoch > 0

    def _fetch_as_overriders(self, info):
        self.targets = Targets(info)
        for o in self.nets[0].overriders:
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
        self.retrain_cnt = 0
        self.target_layer = None
        self.loss_avg = None
        self.best_ckpt = None
        info = self.config.retrain.parameters
        # define initial scale
        if self.retrain_mode == 'overriders':
            self._fetch_as_overriders(info)
        else:
            self._fetch_as_variables(info)
        # run_status = self.config.retrain.run_status
        # self.info = Info(
        #     info, self.tf_session, self.targets.show_targets(), run_status)

    def once(self):
        train_op = self._train_op
        if self.config.retrain.get('eval_only', False):
            # do not run training operations when `retrain.eval_only` is set
            train_op = train_op['imgs_seen']
        tasks = [train_op, self.loss, self.accuracy, self.num_epochs]
        noop, loss, acc, num_epochs = self.run(tasks, update_progress=True)
        if math.isnan(loss):
            raise ValueError('Model diverged with a nan-valued loss.')
        return (loss, acc, num_epochs)

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

    def profile_associated_vars(self, start=False):
        '''
        1. profile the associated vars and determine a priority list
        2. produce a cont list to determine which targeting vars continues on
        retraining
        '''
        if start:
            self.best_ckpt = self.config.system.checkpoint.load
            # if yaml exists, load it and compute self.cont
            if self.config.retrain.get('cont_list'):
                doct_cont = self.config.retrain.cont_list
                self.targets.cont_list(doct_cont)
            # yaml does not exist, initialize to true by default
            else:
                self.targets.cont_list()
        # pick next target
        self.target_layer = self.targets.pick_layer(self, start)

    def _prepare_yaml(self, write):
        name = self.config.model.name
        self.stream = open(
            'trainers/{}_retrain_base.yaml'.format(name), 'w')
        self.dump_data = {
            'retrain': {
                'train_acc_base': float(self.acc_base),
                'loss_base': float(self.loss_base)}}
        if write:
            self.stream.write(yaml.dump(
                self.dump_data,
                default_flow_style=False))
        return

    def profile_for_one_epoch(self):
        epoch = 0
        self._reset_stats()
        self.reset_num_epochs()

        if self.config.retrain.get('train_acc_base'):
            # if acc is hand coded in yaml
            self.acc_base = self.config.retrain.train_acc_base
            log.info(
                'loaded profiled acc baseline, acc is {}'
                .format(self.acc_base))
            if self.config.retrain.get('loss_base'):
                self.loss_base = self.config.retrain.loss_base
            self._prepare_yaml(False)
            return

        tolerance = self.config.retrain.tolerance
        log.info('profiling baseline')
        imgs_seen = self._train_op['imgs_seen']
        tasks = [imgs_seen, self.loss, self.accuracy, self.num_epochs]
        while epoch < 1.0:
            _, loss, acc, epoch = self.run(tasks, update_progress=True)
            self.loss_total += loss
            self.acc_total += acc
            self.step += 1
        self.loss_base = self.loss_total / float(self.step) * (1 + tolerance)
        self.acc_base = self.acc_total / float(self.step) * (1 - tolerance)

        log.info('profiled baseline, loss is {}, acc is {}'.format(
            self.loss_base, self.acc_base))
        self._reset_stats()
        self.reset_num_epochs()
        self._prepare_yaml(True)

    def _avg_stats(self):
        self.loss_avg = self.loss_total / float(self.step)
        self.acc_avg = self.acc_total / float(self.step)
        self._reset_stats()

    def _update_stats(self, loss, acc):
        self.step += 1
        self.loss_total += loss
        self.acc_total += acc

    def _reset_stats(self):
        self.step = 0
        self.loss_total = 0
        self.acc_total = 0

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


class GlobalRetrain(RetrainBase):
    def forward_policy(self, floor_epoch):
        self.best_ckpt = 'retrain-' + str(self.retrain_cnt) + '-' \
            + str(floor_epoch)
        self.save_checkpoint(self.best_ckpt)
        self._cp_epoch = floor_epoch
        self.retrain_cnt += 1
        self.log_thresholds(self.loss_avg, self.acc_avg)
        self.profile_associated_vars()
        self.variables_refresh()
        self.reset_num_epochs()
        threshold = self.targets.members[0].thresholds[0]
        log.info('update threshold to {}, working on {}'.format(
            threshold, self.target_layer))
        return True

    def backward_policy(self):
        # if did not reach min scale
        tmp_tv = self.targets.members[0]
        if tmp_tv.scale > 0:
            scale_check = self._fetch_scale() > tmp_tv.min_scale
            threshold_check = tmp_tv.end_thresholds[0] > tmp_tv.thresholds[0]
        elif tmp_tv.scale == 0:
            scale_check = False
            threshold_check = False
        else:
            scale_check = self._fetch_scale() < tmp_tv.min_scale
            threshold_check = tmp_tv.end_thresholds[0] < tmp_tv.thresholds[0]

        if scale_check and threshold_check:
            # retrace the best ckpt
            self.load_checkpoint(self.best_ckpt)
            self._decrease_scale()
            log.info(
                'Decreasing scale to {}, threshold is {}...'
                .format(self._fetch_scale(), tmp_tv.thresholds[0]))
            self.reset_num_epochs()
            return True
        # stop if reach min scale
        else:
            self.targets.cont = []
            thresholds = tmp_tv.thresholds[0]
            scale = tmp_tv.scale
            log.info(
                'All layers done, final threshold is {}'
                .format(thresholds - scale))
            if not threshold_check:
                log.info('threshold meets its minimum')
            if not scale_check:
                log.info('scale meets its minimum')
            log.info(
                'Overrider is done, model stored at {}.'
                .format(self.best_ckpt))
            self.reset_num_epochs()
            return False

    def variables_refresh(self):
        update_flag = False
        for item in self.targets.members:
            self.variable_refresh(item)
            if isinstance(item.av, OverriderBase):
                item.av.should_update = True
                update_flag = True
        tmp = self.targets.show_associates()[0]
        # check if it is and only is floating point
        check_shift = self._check_cls(tmp, ShiftQuantizer)
        check_float = self._check_cls(tmp, FloatingPointQuantizer) \
            and not check_shift
        # in global retrain, thresholds are globally the same
        w = self.targets.members[0].thresholds[0]
        if check_float:
            self.allocate_exp_mantissa(w)
        if check_shift:
            for item in self.targets.members:
                av_before = self.run(item.av.before)
                if not item.mean_quantizer:
                    raise ValueError('Mean quantizer not defined!')
                self._update_mean_quantizer(av_before, item.mean_quantizer, w)
        if update_flag:
            self.overriders_update()

    def _check_cls(self, av, cls):
        if isinstance(av, cls):
            return True
        if isinstance(av, Recentralizer):
            if isinstance(av.quantizer, cls):
                return True
        return False

    def allocate_exp_mantissa(self, width, overflow_rate=0.0):
        '''
        Description:
            a hack and a specialized method for floating point quantizer.
        Problem:
            floating point quantization has two searching parameters: exponent
            width and mantissa width, each of them has a different impact on
            accuracy.
        Solution:
            I proposed a before retraining loss check. Brute-froce search all
            the possible combinations between exponents and mantissa, pick
            the combination that has the least loss compared to full-precision
            weights
        '''
        log.info("search to allocate exp and mantissa parts")
        biases = losses = []
        for mantissa_width in range(0, int(width) + 1):
            loss = 0
            biases = []
            exponent_width = width - mantissa_width
            for item in self.targets.members:
                org_matrix = self.run(item.av.quantizer.before)
                tmp, bias = item.av.quantizer.compute_quantization_loss(
                    org_matrix[org_matrix != 0], exponent_width,
                    mantissa_width, overflow_rate)
                # accumulate loss
                loss += tmp
                # collect bias
                biases.append(bias)
            losses.append((loss, exponent_width, mantissa_width, biases))
        # find smallest loss
        _, exp_width, mantissa_width, biases = min(
            losses, key=lambda meta: meta[0])
        index = 0
        for item in self.targets.members:
            values = self.run(item.av.before)
            self._update_mean_quantizer(
                values, item.mean_quantizer, width, overflow_rate)
            self.assign(item.quantizer.mantissa_width, mantissa_width)
            self.assign(item.quantizer.exponent_bias, biases[index])
            self.assign(item.bias_quantizer.mantissa_width, mantissa_width)
            self.assign(item.bias_quantizer.exponent_bias, biases[index])
            index += 1

    def _compute_mean_exp(self, pos_mean, neg_mean, width, overflow_rate):
        max_exponent = int(2 ** width)
        for exp in range(min(-max_exponent, -10), max(max_exponent, 4)):
            max_value = 2 ** (exp + 1)
            if neg_mean > -max_value and pos_mean < max_value:
                break
        return exp

    def _update_mean_quantizer(
            self, values, variable, width, overflow_rate=0.01):
        positives = np.mean(values[values > 0])
        negatives = np.mean(values[values < 0])
        exp = self._compute_mean_exp(
            positives, negatives, width, overflow_rate)
        # shift quantizer has mantissa zero
        if not isinstance(variable, ShiftQuantizer):
            if exp > 0:
                self.assign(
                    variable.mantissa_width,
                    width - math.ceil(math.log(exp, 2)))
            else:
                self.assign(variable.mantissa_width, width)
        if exp < 0:
            self.assign(variable.exponent_bias, -exp)
        else:
            self.assign(variable.exponent_bias, 0)

    def _decrease_scale(self):
        # decrease scale factor, for quantizer, this factor might be 1
        for item in self.targets.members:
            # roll back on thresholds
            threshold = item.thresholds[0]
            scale = item.scale
            if threshold == item.start_threshold:
                raise ValueError(
                    'Thresholds failed on starting point, consider '
                    'changing starting point.')
            else:
                item.thresholds = [
                    th - scale for th in item.thresholds]
            # decrease scale
            if isinstance(scale, int) and isinstance(threshold, int):
                item.scale = int(item.scale * item.update_factor)
            else:
                item.scale = item.scale * item.update_factor
        tmp = self.targets.members[0].av
        check_floating_point = self._check_cls(tmp, FloatingPointQuantizer) \
            and not self._check_cls(tmp, ShiftQuantizer)
        for item in self.targets.members:
            # use new scale
            self.variable_refresh(item)
        if check_floating_point:
            w = self.targets.members[0].thresholds[0]
            self.allocate_exp_mantissa(w)
        self.overriders_update()


class LayerwiseRetrain(RetrainBase):
    def forward_policy(self, floor_epoch):
        log.debug('Targeting on {}...'.format(self.target_layer.name))
        log.debug('Log: {}'.format(self.log))
        self.best_ckpt = 'retrain-' + str(self.retrain_cnt) + '-' \
            + str(floor_epoch)
        self.save_checkpoint(self.best_ckpt)
        self._cp_epoch = floor_epoch
        self.retrain_cnt += 1
        self.log_thresholds(self.loss_avg, self.acc_avg)
        self.profile_associated_vars()
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
                # fetch a new layer to retrain
                self.profile_associated_vars()
                self.variables_refresh()
                self.reset_num_epochs()
                if not threshold_check:
                    log.info('threshold meets its minimum')
                if not scale_check:
                    log.info('scale meets its minimum')
                log.info('switching layer, working on {}'.format(
                    self.target_layer.name))
                log.info('priority_list: {}'.format(self.targets.priority_list))
                # refresh the yaml
                self.dump_data['retrain']['cont_list'] = self.targets.cont
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
