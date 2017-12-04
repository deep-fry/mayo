import math
import numpy as np
import tensorflow as tf
import yaml
import pickle
import operator

from mayo.log import log
from mayo.session.train import Train
from mayo.override.base import OverriderBase, ChainOverrider
from mayo.override.quantize import Recentralizer, FloatingPointQuantizer
from mayo.override.quantize import ShiftQuantizer


class Info(object):
    def __init__(self, meta_info, session, targeting_vars, associated_vars,
        run_status):
        self.scales = {}
        self.min_scales = {}
        self.scale_update_factors = {}
        self.start_ths = {}
        self.ths = {}
        self.max_ths = {}

        # now only supports retraining on one overrider
        self.meta = meta_info
        self.targeting_vars = targeting_vars
        self.associated_vars = associated_vars

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


class RetrainBase(Train):
    def retrain(self):
        log.debug('Retraining start.')
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

    def _check_overrider_type(self, overrider, o_type):
        o_name = overrider.__class__.__name__
        if isinstance(o_type, list):
            for ot in o_type:
                if o_name in ot:
                    return True
        if o_name in o_type:
            return True
        return False

    def _fetch_as_overriders(self, info):
        for o in self.nets[0].overriders:
            # fetch the targeting variables
            # handel chained overrider
            if isinstance(o, ChainOverrider):
                for chained_o in o:
                    if self._check_overrider_type(chained_o, info.type):
                        o = chained_o
            if isinstance(o, Recentralizer):
                if self._check_overrider_type(chained_o, info.type):
                    # a temporary hack
                    self.targeting_vars.append(getattr(
                        o.mean_quantizer,
                        info.target))
                    self.associated_vars.append(o)
                    o = o.quantizer
            self.targeting_vars.append(getattr(o, info.target))
            self.associated_vars.append(o)

    def _fetch_as_variables(self, info):
        for name in info.target:
            for item in self.global_variables():
                if name in item.name:
                    self.targeting_vars.append(item)
        for name in info.associated:
            for item in self.global_variables():
                if name in item.name:
                    self.associated_vars.append(item)

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
        info = self.config.retrain.parameters
        self.targeting_vars = []
        self.associated_vars = []
        # define initial scale
        if self.retrain_mode == 'overriders':
            self._fetch_as_overriders(info)
        else:
            self._fetch_as_variables(info)
        run_status = self.config.retrain.run_status
        self.info = Info(info, self.tf_session, self.targeting_vars,
            self.associated_vars, run_status)

    def _init_retrain(self):
        self.retrain_mode = self.config.retrain.retrain_mode
        self._init_scales()
        self._reset_stats()
        self._reset_vars()

        self.profile_associated_vars(start=True)
        self.profile_for_one_epoch()
        # init all overriders
        for variable in self.targeting_vars:
            self.variable_init(variable)

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

    def _retrain_iteration(self):
        system = self.config.system
        loss, acc, epoch = self.once()
        self._update_stats(loss, acc)
        summary_delta = self.change.delta('summary.epoch', epoch)

        if system.summary.save and summary_delta >= 0.1:
            self._save_summary(epoch)
        floor_epoch = math.floor(epoch)
        cp_interval = system.checkpoint.get('save.interval', 0)

        # if epoch > 0.1:
        if self.change.every('checkpoint.epoch', floor_epoch, cp_interval):
            self._avg_stats()
            if self.acc_avg >= self.acc_base:
                return self.forward_policy(floor_epoch)

            iter_max_epoch = self.config.retrain.iter_max_epoch

            # current setup exceeds max epoches, retrieve backwards
            if epoch >= iter_max_epoch and epoch > 0:
                self.retrain_cnt += 1
                self.reset_num_epochs()
                self.log_thresholds(self.loss_avg, self.acc_avg)
                return self.backward_policy()
        return True

    def backward_policy(self):
        raise NotImplementedError(
            'Method of backward policy is not implemented.')

    def forward_policy(self, floor_epoch):
        raise NotImplementedError(
            'Method of forward policy is not implemented.')

    def log_thresholds(self, loss, acc):
        raise NotImplementedError(
            'Method of logging threholds is not implemented.')

    def _fetch_scale(self):
        for tv in self.targeting_vars:
            if tv.name == self.target_layer:
                return self.info.get(tv, 'scale')

    def profile_associated_vars(self, start=False):
        self.priority_list = []
        if start:
            name = self.config.system.checkpoint.load
            self.best_ckpt = name
            self.cont = {}
            if self.config.retrain.get('cont_list'):
                # if yaml exists, load it and compute self.cont
                doct_cont = self.config.retrain.cont_list
                for variable in self.targeting_vars:
                    if doct_cont.get(variable.name):
                        self.cont[variable.name] = True
                    else:
                        self.cont[variable.name] = False
            else:
                for variable in self.targeting_vars:
                    name = variable.name
                    self.cont[name] = True
        d = {}
        thresholds = {}
        scales = {}
        for tv, av in zip(self.targeting_vars, self.associated_vars):
            name = tv.name
            d[name] = self._metric_clac(av)
            thresholds[name] = self.info.get(tv, 'threshold')
            scales[name] = self.info.get(tv, 'scale')

        for key in sorted(d, key=d.get):
            log.debug('key is {} cont is {}'.format(key, self.cont[key]))
            if self.cont[key]:
                self.priority_list.append(key)
        log.debug('display layerwise metric')
        log.debug('{}'.format(d))
        log.debug('display thresholds')
        log.debug('{}'.format(thresholds))
        log.debug('display scales')
        log.debug('{}'.format(scales))
        log.debug('display priority list info')
        log.debug('{}'.format(self.priority_list))
        log.debug('stored checkpoint')
        log.debug('{}'.format(self.best_ckpt))
        if self.priority_list == []:
            log.debug('list is empty!!')
        else:
            self.target_layer = self.priority_list.pop()

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
        log.info('Start profiling for one epoch')
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
        self._reset_stats()
        self.reset_num_epochs()
        log.info('profiled baseline, loss is {}, acc is {}'.format(
            self.loss_base,
            self.acc_base,
        ))
        self._reset_stats()
        self._prepare_yaml(True)

    def _metric_clac(self, variable):
        if isinstance(variable, (tf.Variable, tf.Tensor)):
            return self.run(variable).size
        if isinstance(variable, ChainOverrider):
            chained_vars = [self._find_metric(v) for v in variable]
            metric_value = reduce(operator.mul, chained_vars)
        if isinstance(variable, OverriderBase):
            metric_value = self._find_metric(variable)
        # normal tensor value should have been returned
        return metric_value * self.run(variable.after).size

    def _find_metric(self, value):
        if hasattr(value, 'width'):
            return self.run(value.width)
        if hasattr(value, 'mask'):
            density = np.count_nonzero(self.run(value.mask)) / \
                float(self.run(value.after).size)
            return density
        else:
            return self.run(value.after).size

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

    def _reset_vars(self):
        self.log = {}
        self.retrain_cnt = 0

        self.target_layer = None
        self.loss_avg = None
        self.best_ckpt = None

    def variables_refresh(self):
        raise NotImplementedError(
            'Method to refresh overriders is not implemented.')

    def variable_init(self, v):
        threshold = self.info.get(v, 'threshold')
        self.assign(v, threshold)

    def variable_refresh(self, v):
        threshold = self.info.get(v, 'threshold')
        scale = self.info.get(v, 'scale')
        self.info.set(v, 'threshold', threshold + scale)
        self.assign(v, threshold + scale)


class GlobalRetrain(RetrainBase):
    def variables_refresh(self):
        update_flag = False
        for tv, av in zip(self.targeting_vars, self.associated_vars):
            self.variable_refresh(tv)
            if isinstance(av, OverriderBase):
                av.should_update = True
                update_flag = True
            if self.config.retrain.parameters.get('update_overrider'):
                for o in self.nets[0].overriders:
                    if o.name in av.name:
                        o.should_update = True
                        update_flag = True
        check_floating_point = isinstance(self.associated_vars[0],
            FloatingPointQuantizer) and not isinstance(self.associated_vars[0],
            ShiftQuantizer)
        if check_floating_point:
            w = self.info.get(self.targeting_vars[0], 'threshold')
            self.allocate_exp_mantissa(w)
        if isinstance(self.associated_vars[0], Recentralizer) and \
            check_floating_point:
            w = self.info.get(self.targeting_vars[0], 'threshold')
            self.allocate_exp_mantissa(w)
        if update_flag:
            self.overriders_update()

    def allocate_exp_mantissa(self, width, overflow_rate=0.01):
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
        for mantissa_width in range(1, int(width)):
            loss = 0
            biases = []
            exponent_width = width - mantissa_width
            for av in self.associated_vars:
                if not isinstance(av, Recentralizer):
                    tmp, bias = av.compute_quantization_loss(
                        self.run(av.before), exponent_width, mantissa_width,
                        overflow_rate)
                    # accumulate loss
                    loss += tmp
                    # collect bias
                    biases.append(bias)
            losses.append((loss, exponent_width, mantissa_width, biases))
        # find smallest loss
        _, exp_width, mantissa_width, biases = min(losses,
            key=lambda meta: meta[0])
        index = 0
        for av in self.associated_vars:
            if isinstance(av, Recentralizer):
                values = self.run(av.before)
                positives = np.mean(values > 0)
                negatives = np.mean(values < 0)
                exp = av.mean_quantizer.compute_mean_exp(positives, negatives,
                        width, overflow_rate)
                av = av.mean_quantizer
                self.assign(av.mantissa_width, width - 1)
                av.exponent_bias = exp
            else:
                self.assign(av.mantissa_width, mantissa_width)
                av.exponent_bias = biases[index]
                index += 1

    def forward_policy(self, floor_epoch):
        self.save_checkpoint(
            'th-' + str(self.retrain_cnt) + '-' + str(floor_epoch))
        self.best_ckpt = 'th-' + str(self.retrain_cnt) + '-' \
            + str(floor_epoch)
        self._cp_epoch = floor_epoch
        self.retrain_cnt += 1
        self.log_thresholds(self.loss_avg, self.acc_avg)
        self.profile_associated_vars()
        self.variables_refresh()
        self.reset_num_epochs()
        for tv in self.targeting_vars:
            if tv.name == self.target_layer:
                threshold = self.info.get(tv, 'threshold')
                log.info('update threshold to {}, working on {}'.format(
                    threshold, self.target_layer))
        return True

    def log_thresholds(self, loss, acc):
        _, _, prev_loss = self.log.get(self.target_layer, [None, None, None])
        for tv in self.targeting_vars:
            name = tv.name
            value = self.info.get(tv, 'threshold')
            if prev_loss is None:
                self.log[name] = (value, loss, acc)
            else:
                if acc > self.acc_base:
                    self.log[self.target_layer] = (value, loss, acc)

    def backward_policy(self):
        # if did not reach min scale
        tmp_tv = self.targeting_vars[0]
        end_scale = self.info.get(tmp_tv, 'end_scale')
        scale = self.info.get(tmp_tv, 'scale')
        end_threshold = self.info.get(tmp_tv, 'end_threshold')
        threshold = self.info.get(tmp_tv, 'threshold')
        if scale >= 0:
            scale_check = self._fetch_scale() > end_scale
            threshold_check = end_threshold > threshold
            run = scale_check and threshold_check
        else:
            scale_check = self._fetch_scale() < end_scale
            threshold_check = end_threshold < threshold
            run = scale_check and threshold_check

        if run:
            # retrace the best ckpt
            self.load_checkpoint(self.best_ckpt)
            self._decrease_scale()
            thresholds = self.info.get(tmp_tv, 'threshold')
            log.info(
                'Decreasing scale to {}, threshold is {}...'
                .format(self._fetch_scale(), thresholds))
            self.reset_num_epochs()
            return True
        # stop if reach min scale
        else:
            self.cont[self.target_layer] = False
            thresholds = self.info.get(tmp_tv, 'threshold')
            scale = self.info.get(tmp_tv, 'scale')
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

    def _decrease_scale(self):
        # decrease scale factor, for quantizer, this factor might be 1
        for tv in self.targeting_vars:
            # roll back on thresholds
            threshold = self.info.get(tv, 'threshold')
            scale = self.info.get(tv, 'scale')
            if threshold == self.info.get(tv, 'start_threshold'):
                raise ValueError(
                    'Threshold failed on starting point, consider '
                    'changing your starting point.')
            else:
                self.info.set(tv, 'threshold', threshold - scale)
            # decrease scale
            factor = self.info.get(tv, 'scale_factor')
            if isinstance(scale, int) and isinstance(threshold, int):
                self.info.set(tv, 'scale', int(scale * factor))
            else:
                self.info.set(tv, 'scale', scale * factor)
            # use new scale
            self.variable_refresh(tv)
        if isinstance(self.associated_vars[0], FloatingPointQuantizer):
            w = self.info.get(self.targeting_vars[0], 'threshold')
            self.allocate_exp_mantissa(w)


class LayerwiseRetrain(RetrainBase):
    def variables_refresh(self):
        for tv, av in zip(self.targeting_vars, self.associated_vars):
            if tv.name == self.target_layer:
                self.variable_refresh(tv)
                if isinstance(av, OverriderBase):
                    av.should_update = True
        if isinstance(av, OverriderBase):
            self.overriders_update()

    def forward_policy(self, floor_epoch):
        log.debug('Targeting on {}...'.format(self.target_layer))
        log.debug('Log: {}'.format(self.log))
        self.save_checkpoint(
            'th-' + str(self.retrain_cnt) + '-' + str(floor_epoch))
        self.best_ckpt = 'th-' + str(self.retrain_cnt) + '-' \
            + str(floor_epoch)
        self._cp_epoch = floor_epoch
        self.retrain_cnt += 1
        self.log_thresholds(self.loss_avg, self.acc_avg)
        self.profile_associated_vars()
        self.variables_refresh()
        self.reset_num_epochs()
        for tv in self.targeting_vars:
            if tv.name == self.target_layer:
                threshold = self.info.get(tv, 'threshold')
        log.info(
            'update threshold to {}, working on {}'
            .format(threshold, self.target_layer))
        return True

    def backward_policy(self):
        finished = not self.cont[self.target_layer]
        if self.priority_list == [] and finished:
            log.info(
                'Overrider is done, model stored at {}.'
                .format(self.best_ckpt))
            for tv in self.targeting_vars:
                threshold = self.info.get(tv, 'threshold')
                scale = self.info.get(tv, 'scale')
                log.info(
                    'Layer name: {}, threshold: {}, scale: {}.'
                    .format(tv.name, threshold, scale))
            return False
        else:
            # trace back
            self.load_checkpoint(self.best_ckpt)
            # current layer is done
            for tv in self.targeting_vars:
                if tv.name == self.target_layer:
                    recorded = tv
                    break
            end_scale = self.info.get(recorded, 'end_scale')
            scale = self.info.get(recorded, 'scale')
            end_threshold = self.info.get(recorded, 'end_threshold')
            threshold = self.info.get(recorded, 'threshold')
            if scale >= 0:
                scale_check = self._fetch_scale() > end_scale
                threshold_check = end_threshold > threshold
            else:
                scale_check = self._fetch_scale() < end_scale
                threshold_check = end_threshold < threshold
            run = scale_check and threshold_check
            if run:
                # overriders are refreshed inside decrease scale
                self._decrease_scale()
                self.reset_num_epochs()
                log.info(
                    'Decreasing scale to {}, working on {}...'
                    .format(self._fetch_scale(), self.target_layer))
            else:
                # threshold roll back
                threshold = self.info.get(recorded, 'threshold')
                scale = self.info.get(recorded, 'scale')
                self.info.set(recorded, 'threshold', threshold - scale)
                self.cont[self.target_layer] = False
                # fetch a new layer to retrain
                self.profile_associated_vars()
                self.variables_refresh()
                self.reset_num_epochs()
                if not threshold_check:
                    log.info('threshold meets its minimum')
                if not scale_check:
                    log.info('scale meets its minimum')
                log.info('switching layer, working on {}'.format(
                    self.target_layer))
                log.info('priority_list: {}'.format(self.priority_list))
                # refresh the yaml
                self.dump_data['retrain']['cont_list'] = self.cont
                yaml.dump(
                    self.dump_data, self.stream, default_flow_style=False)
            return True

    def _decrease_scale(self):
        # decrease scale factor, for quantizer, this factor might be 1
        for tv in self.targeting_vars:
            if tv.name == self.target_layer:
                # threshold roll back
                threshold = self.info.get(tv, 'threshold')
                scale = self.info.get(tv, 'scale')
                if threshold == self.info.get(tv, 'start_threshold'):
                    raise ValueError(
                        'Threshold failed on starting point, consider '
                        'changing your starting point.')
                else:
                    self.info.set(tv, 'threshold', threshold - scale)
                factor = self.info.get(tv, 'scale_factor')
                # update scale
                if isinstance(scale, int) and isinstance(threshold, int):
                    self.info.set(tv, 'scale', int(scale * factor))
                else:
                    self.info.set(tv, 'scale', scale * factor)
                self.variable_refresh(tv)

    def log_thresholds(self, loss, acc):
        _, _, prev_loss = self.log.get(self.target_layer, [None, None, None])
        for tv in self.targeting_vars:
            if tv.name == self.target_layer:
                value = self.info.get(tv, 'threshold')
                break
        if prev_loss is None:
            self.log[self.target_layer] = (value, loss, acc)
        else:
            if acc > self.acc_base:
                self.log[self.target_layer] = (value, loss, acc)
