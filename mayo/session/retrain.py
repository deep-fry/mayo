import math
import numpy as np
import tensorflow as tf
import yaml

from mayo.log import log
from mayo.session.train import Train


class OverriderInfo(object):
    def __init__(self, overriders_info, overriders, session):
        self.max_ths = {}
        self.min_scales = {}
        self.ths = {}
        self.scales = {}
        self.start_ths = {}
        self.scale_update_factors = {}
        self.targets = {}
        # now only supports retraining on one overrider
        self.meta = overriders_info[0]

        for meta in overriders_info:
            scale_dict = {}
            update_dict = {}
            th_dict = {}
            th_max_dict = {}
            th_start_dict = {}
            scale_min_dict = {}
            for o in overriders:
                # overrider might be wrapped in ChainOverrider
                o = self.get_overrider(o)
                if o is not None:
                    scale_dict[o.name] = meta.range['scale']
                    scale_min_dict[o.name] = meta.range['min_scale']
                    update_dict[o.name] = meta.scale_update_factor
                    if meta.special == 'continue':
                        if not hasattr(o, meta.target):
                            raise ValueError(
                                'Missing {} in checkpoint.'
                                .format(meta.target))
                        th = session.run(getattr(o, meta.target))
                        th_dict[o.name] = th
                        th_start_dict[o.name] = th
                        log.info('{} is continuing on {}.'.format(o.name, th))
                    else:
                        th_dict[o.name] = meta.range['from']
                        th_start_dict[o.name] = meta.range['from']
                    th_max_dict[o.name] = meta.range['to']
                    cls_name = o.__class__.__name__
            if scale_dict == {}:
                raise ValueError(
                    '{} is not found in overrider definitions, '
                    'but has been specified as a target.'.format(meta.type))
            self.scales[cls_name] = scale_dict
            self.min_scales[cls_name] = scale_min_dict
            self.ths[cls_name] = th_dict
            self.start_ths[cls_name] = th_start_dict
            self.max_ths[cls_name] = th_max_dict
            self.scale_update_factors[cls_name] = update_dict
            self.targets[cls_name] = str(meta.target)

    def get(self, overrider, info_type):
        overrider = self.get_overrider(overrider)
        cls_name = overrider.__class__.__name__
        if info_type == 'end_threshold':
            return self.max_ths[cls_name][overrider.name]
        if info_type == 'end_scale':
            return self.min_scales[cls_name][overrider.name]
        if info_type == 'threshold':
            return self.ths[cls_name][overrider.name]
        if info_type == 'start_threshold':
            return self.start_ths[cls_name][overrider.name]
        if info_type == 'scale':
            return self.scales[cls_name][overrider.name]
        if info_type == 'scale_factor':
            return self.scale_update_factors[cls_name][overrider.name]
        if info_type == 'target':
            return self.targets[cls_name]
        raise ValueError('{} is not a collected info key.'.format(info_type))

    def set(self, overrider, info_type, value):
        overrider = self.get_overrider(overrider)
        cls_name = overrider.__class__.__name__
        if info_type == 'threshold':
            self.ths[cls_name][overrider.name] = value
        elif info_type == 'scale':
            self.scales[cls_name][overrider.name] = value
        else:
            raise ValueError(
                '{} is not a collected info key.'.format(info_type))

    def get_overrider(self, overrider):
        cls_name = overrider.__class__.__name__
        if cls_name == 'ChainOverrider':
            for chained_overrider in overrider:
                if chained_overrider.__class__.__name__ in self.meta.type:
                    return chained_overrider
            return None
        if cls_name in self.meta.type:
            return overrider
        return None


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

    def _init_scales(self):
        # define initial scale
        overriders_info = self.config.retrain.overriders
        self.info = OverriderInfo(
            overriders_info, self.nets[0].overriders, self.tf_session)

    def _init_retrain(self):
        self._init_scales()
        self._reset_stats()
        self._reset_vars()

        self.profile_overrider(start=True)
        self.profile_for_one_epoch()
        # init all overriders
        for o in self.nets[0].overriders:
            self.overrider_init(o)

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
        for o in self.nets[0].overriders:
            if o.name == self.target_layer:
                return self.info.get(o, 'scale')

    def profile_overrider(self, start=False):
        self.priority_list = []
        if start:
            name = self.config.system.checkpoint.load
            self.best_ckpt = name
            self.cont = {}
            if self.config.retrain.get('cont_list'):
                # if yaml exists, load it and compute self.cont
                doct_cont = self.config.retrain.cont_list
                for o in self.nets[0].overriders:
                    if doct_cont.get(o.name):
                        self.cont[o.name] = True
                    else:
                        self.cont[o.name] = False
            else:
                for o in self.nets[0].overriders:
                    name = o.name
                    self.cont[name] = True
        d = {}
        thresholds = {}
        scales = {}
        for o in self.nets[0].overriders:
            name = o.name
            d[name] = self._metric_clac(o)
            thresholds[name] = self.info.get(o, 'threshold')
            scales[name] = self.info.get(o, 'scale')

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
            name = self.config.model.name
            self.stream = open(
                'trainers/{}_retrain_base.yaml'.format(name), 'w')
            self.dump_data = {
                'retrain': {
                    'train_acc_base': float(self.acc_base),
                    'loss_base': float(self.loss_base),
                },
            }
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
        name = self.config.model.name
        self.stream = open(
            'trainers/log/{}_retrain_base.yaml'.format(name), 'w')
        self.dump_data = {
            'retrain': {
                'train_acc_base': float(self.acc_base),
                'loss_base': float(self.loss_base),
            },
        }
        yaml.dump(self.dump_data, self.stream, default_flow_style=False)

    def _metric_clac(self, o):
        metric_value = num_elements = self.run(o.after).size
        if hasattr(o, '_mask'):
            valid_elements = np.count_nonzero(self.run(o._mask))
            density = valid_elements / float(num_elements)
            metric_value *= density
        if hasattr(o, 'width'):
            if isinstance(o.width, (tf.Variable, tf.Tensor)):
                bits = self.run(o.width)
            else:
                bits = o.width
            metric_value *= bits
        return metric_value

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

    def overriders_refresh(self):
        raise NotImplementedError(
            'Method to refresh overriders is not implemented.')

    def overrider_init(self, o):
        threshold = self.info.get(o, 'threshold')
        target = self.info.get(o, 'target')
        o._parameter_variables_assignment[target] = threshold

    def overrider_refresh(self, o):
        threshold = self.info.get(o, 'threshold')
        scale = self.info.get(o, 'scale')
        target = self.info.get(o, 'target')
        self.info.set(o, 'threshold', threshold + scale)
        setattr(o, target, threshold + scale)


class GlobalRetrain(RetrainBase):
    def overriders_refresh(self):
        for o in self.nets[0].overriders:
            self.overrider_refresh(o)
            o.should_update = True
        self.overriders_update()

    def forward_policy(self, floor_epoch):
        self.save_checkpoint(
            'th-' + str(self.retrain_cnt) + '-' + str(floor_epoch))
        self.best_ckpt = 'th-' + str(self.retrain_cnt) + '-' \
            + str(floor_epoch)
        self._cp_epoch = floor_epoch
        self.retrain_cnt += 1
        self.log_thresholds(self.loss_avg, self.acc_avg)
        self.profile_overrider()
        self.overriders_refresh()
        self.reset_num_epochs()
        for o in self.nets[0].overriders:
            if o.name == self.target_layer:
                threshold = self.info.get(o, 'threshold')
        log.info('update threshold to {}, working on {}'.format(
            threshold, self.target_layer))
        return True

    def log_thresholds(self, loss, acc):
        _, _, prev_loss = self.log.get(self.target_layer, [None, None, None])
        for o in self.nets[0].overriders:
            value = self.info.get(o, 'threshold')
            if prev_loss is None:
                self.log[o.name] = (value, loss, acc)
            else:
                if acc > self.acc_base:
                    self.log[self.target_layer] = (value, loss, acc)

    def backward_policy(self):
        # if did not reach min scale
        end_scale = self.info.get(self.nets[0].overriders[0], 'end_scale')
        scale = self.info.get(self.nets[0].overriders[0], 'scale')
        if scale >= 0:
            should_continue = self._fetch_scale() > end_scale
        else:
            should_continue = self._fetch_scale() < end_scale

        if should_continue:
            # retrace the best ckpt
            self.load_checkpoint(self.best_ckpt)
            self._decrease_scale()
            thresholds = self.info.get(self.nets[0].overriders[0], 'threshold')
            log.info(
                'Decreasing scale to {}, threshold is {}...'
                .format(self._fetch_scale(), thresholds))
            self.reset_num_epochs()
            return True
        # stop if reach min scale
        else:
            for o in self.nets[0].overriders:
                self.cont[self.target_layer] = False
            thresholds = self.info.get(self.nets[0].overriders[0], 'threshold')
            log.info(
                'All layers done, final threshold is {}'
                .format(thresholds))
            self.reset_num_epochs()
            return False

    def _decrease_scale(self):
        # decrease scale factor, for quantizer, this factor might be 1
        for o in self.nets[0].overriders:
            # roll back on thresholds
            threshold = self.info.get(o, 'threshold')
            scale = self.info.get(o, 'scale')
            if threshold == self.info.get(o, 'start_threshold'):
                raise ValueError(
                    'Threshold failed on starting point, consider '
                    'changing your starting point.')
            else:
                self.info.set(o, 'threshold', threshold - scale)
            # decrease scale
            factor = self.info.get(o, 'scale_factor')
            if isinstance(scale, int) and isinstance(threshold, int):
                self.info.set(o, 'scale', int(scale * factor))
            else:
                self.info.set(o, 'scale', scale * factor)
            # use new scale
            self.overrider_refresh(o)


class LayerwiseRetrain(RetrainBase):
    def overriders_refresh(self):
        for o in self.nets[0].overriders:
            if o.name == self.target_layer:
                self.overrider_refresh(o)
                o.should_update = True
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
        self.profile_overrider()
        self.overriders_refresh()
        self.reset_num_epochs()
        for o in self.nets[0].overriders:
            if o.name == self.target_layer:
                threshold = self.info.get(o, 'threshold')
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
            for o in self.nets[0].overriders:
                threshold = self.info.get(o, 'threshold')
                scale = self.info.get(o, 'scale')
                log.info(
                    'Layer name: {}, threshold: {}, scale: {}.'
                    .format(o.name, threshold, scale))
            return False
        else:
            # trace back
            self.load_checkpoint(self.best_ckpt)
            # current layer is done
            for o in self.nets[0].overriders:
                if o.name == self.target_layer:
                    o_recorded = o
                    break
            end_scale = self.info.get(o_recorded, 'end_scale')
            scale = self.info.get(o_recorded, 'scale')
            if scale >= 0:
                should_continue = self._fetch_scale() > end_scale
            else:
                should_continue = self._fetch_scale() < end_scale
            if should_continue:
                # overriders are refreshed inside decrease scale
                self._decrease_scale()
                self.reset_num_epochs()
                log.info(
                    'Decreasing scale to {}, working on {}...'
                    .format(self._fetch_scale(), self.target_layer))
            else:
                # threshold roll back
                threshold = self.info.get(o_recorded, 'threshold')
                scale = self.info.get(o_recorded, 'scale')
                self.info.set(o_recorded, 'threshold', threshold - scale)
                self.cont[self.target_layer] = False
                # fetch a new layer to retrain
                self.profile_overrider()
                self.overriders_refresh()
                self.reset_num_epochs()
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
        for o in self.nets[0].overriders:
            if o.name == self.target_layer:
                # threshold roll back
                threshold = self.info.get(o, 'threshold')
                scale = self.info.get(o, 'scale')
                if threshold == self.info.get(o, 'start_threshold'):
                    raise ValueError(
                        'Threshold failed on starting point, consider '
                        'changing your starting point.')
                else:
                    self.info.set(o, 'threshold', threshold - scale)
                factor = self.info.get(o, 'scale_factor')
                # update scale
                if isinstance(scale, int) and isinstance(threshold, int):
                    self.info.set(o, 'scale', int(scale * factor))
                else:
                    self.info.set(o, 'scale', scale * factor)
                self.overrider_refresh(o)

    def log_thresholds(self, loss, acc):
        _, _, prev_loss = self.log.get(self.target_layer, [None, None, None])
        for o in self.nets[0].overriders:
            if o.name == self.target_layer:
                value = self.info.get(o, 'threshold')
                break
        if prev_loss is None:
            self.log[self.target_layer] = (value, loss, acc)
        else:
            if acc > self.acc_base:
                self.log[self.target_layer] = (value, loss, acc)
