import math
import numpy as np
import tensorflow as tf

from mayo.log import log
from mayo.session.train import Train


class Overrider_info(object):
    def __init__(self, overriders_info, overriders):
        self.max_ths = {}
        self.min_scales = {}
        self.ths = {}
        self.scales = {}
        self.scale_update_factors = {}
        self.targets = {}

        for meta in overriders_info:
            scale_dict = {}
            update_dict = {}
            th_dict = {}
            th_max_dict = {}
            scale_min_dict = {}
            for o in overriders:
                if o.__class__.__name__ in meta.type:
                    scale_dict[o.name] = meta.range['scale']
                    scale_min_dict[o.name] = meta.range['min_scale']
                    update_dict[o.name] = meta.scale_update_factor
                    th_dict[o.name] = meta.range['from']
                    th_max_dict[o.name] = meta.range['to']
                    cls_name = o.__class__.__name__
            if scale_dict == {}:
                raise ValueError('{} is not found in overrider definitions,'
                    'but has specified as a target'.format(meta.type))
            self.scales[cls_name] = scale_dict
            self.min_scales[cls_name] = scale_min_dict
            self.ths[cls_name] = th_dict
            self.max_ths[cls_name] = th_max_dict
            self.scale_update_factors[cls_name] = update_dict
            self.targets[cls_name] = str(meta.target)

    def get(self, overrider, info_type):
        cls_name = overrider.__class__.__name__
        if info_type == 'end_threshold':
            return self.max_ths[cls_name][overrider.name]
        elif info_type == 'end_scale':
            return self.min_scales[cls_name][overrider.name]
        elif info_type == 'threshold':
            return self.ths[cls_name][overrider.name]
        elif info_type == 'scale':
            return self.scales[cls_name][overrider.name]
        elif info_type == 'scale_factor':
            return self.scale_update_factors[cls_name][overrider.name]
        elif info_type == 'target':
            return self.targets[cls_name]
        else:
            raise ValueError('{} is not a collected info key.'.format(
                info_type))

    def set(self, overrider, info_type, value):
        cls_name = overrider.__class__.__name__
        if info_type == 'threshold':
            self.ths[cls_name][overrider.name] = value
            return
        elif info_type == 'scale':
            self.scales[cls_name][overrider.name] = value
            return
        else:
            raise ValueError('{} is not a collected info key.'.format(
                info_type))



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
        self.info = Overrider_info(overriders_info, self.nets[0].overriders)
        return

    def _init_retrain(self):
        self._init_scales()
        self._reset_stats()
        self._reset_vars()

        self.profile_overrider(start=True)
        self.profile_for_one_epoch()
        # init all overriders
        for o in self.nets[0].overriders:
            self.overrider_init(o)

    def _retrain_iteration(self):
        system = self.config.system
        loss, acc, epoch = self.once()
        self._update_stats(loss, acc)

        if math.isnan(loss):
            raise ValueError("Model diverged with a nan-valued loss")
        self._update_progress(epoch, loss, acc, self._cp_epoch)
        summary_delta = self.change.delta('summary.epoch', epoch)

        if system.summary.save and summary_delta >= 0.1:
            self._save_summary(epoch)
        floor_epoch = math.floor(epoch)
        cp_interval = system.checkpoint.get('save.interval', 0)

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
            self.best_ckpt = 'pretrained'
            self.cont = {}
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
            self.priority_list.append(key)
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

    def empty_eval_run(self):
        tasks = []
        tasks.append(tf.assign_add(self.imgs_seen, self.batch_size))
        tasks.append(self.loss)
        tasks.append(self.accuracy)
        tasks.append(self.num_epochs)
        return tasks

    def profile_for_one_epoch(self):
        log.info('Start profiling for one epoch')
        epoch = 0
        self._reset_stats()
        self.reset_num_epochs()
        if self.config.retrain.get('train_acc_base'):
            # if acc is hand coded in yaml
            self.acc_base = self.config.retrain.train_acc_base
            log.debug('profiled baselines, acc is {}'.format(
                self.acc_base
            ))
            return
        tolerance = self.config.retrain.tolerance
        log.info('profiling baseline')
        # while epoch < 1.0:
        while epoch < 0.1:
            _, loss, acc, epoch = self.run(
                self.empty_eval_run())
            self.loss_total += loss
            self.acc_total += acc
            self.step += 1
            self._update_progress(epoch, loss, acc, self._cp_epoch)
        self.loss_base = self.loss_total / float(self.step) * (1 + tolerance)
        self.acc_base = self.acc_total / float(self.step) * (1 - tolerance)
        self._reset_stats()
        self.reset_num_epochs()
        log.info('profiled baseline, loss is {}, acc is {}'.format(
            self.loss_base,
            self.acc_base,
        ))
        self._reset_stats()

    def _metric_clac(self, o):
        metric_value = num_elements = self.run(o.after).size
        if hasattr(o, '_mask'):
            valid_elements = np.count_nonzero(self.run(o._mask))
            density = valid_elements / float(num_elements)
            metric_value *= density
        if hasattr(o, 'width'):
            # pin the library name to assert
            if tf.__name__ in type(o.width).__module__:
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
        o._parameter_variables_assignment[target] = threshold + scale


class GlobalRetrain(RetrainBase):
    def overriders_refresh(self):
        for o in self.nets[0].overriders:
            self.overrider_refresh(o)
            o.should_update = True
        self.overriders_update()

    def forward_policy(self, floor_epoch):
        log.debug('Targeting on {}'.format(self.target_layer))
        log.debug('log: {}'.format(self.log))
        with log.demote():
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
        if self._fetch_scale() >= self.info.get(self.nets[0].overriders[0],
                                                'end_scale'):
            self._decrease_scale()
            log.debug('recover threholds to {}'.format(
                self._fetch_scale()
            ))
            return True
        # stop if reach min scale
        else:
            for o in self.nets[0].overriders:
                self.cont[self.target_layer] = False
            log.debug('all layers done')
            return False

    def _decrease_scale(self):
        # decrease scale factor, for quantizer, this factor might be 1
        for o in self.nets[0].overriders:
            # roll back on thresholds
            threshold = self.info.get(o, 'threshold')
            scale = self.info.get(o, 'scale')
            self.info.set(o, 'threshold', threshold - scale)
            # decrease scale
            factor = self.info.get(o, 'scale_factor')
            self.info.set(o, 'scale', scale * factor)


class LayerwiseRetrain(RetrainBase):
    def overriders_refresh(self):
        for o in self.nets[0].overriders:
            if o.name == self.target_layer:
                o._threshold_update()
                o.should_update = True
        self.overriders_update()

    def forward_policy(self, floor_epoch):
        log.debug('Targeting on {}'.format(self.target_layer))
        log.debug('log: {}'.format(self.log))
        with log.demote():
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
        return True

    def backward_policy(self):
        finished = self.cont[self.target_layer] is False
        if self.priority_list == [] and finished:
            log.info('overrider is done, model stored at {}'.format(
                self.best_ckpt))
            for o in self.nets[0].overriders:
                log.info('layer name: {}, crate:{}, scale:{}'.format(
                    o.name,
                    getattr(o, self.threshold_name),
                    o.scale))
            return False
        else:
            # current layer is done
            if self._fetch_scale() >= self.config.retrain.min_scale:
                self._decrease_scale()
                log.debug('min scale is {}'.format(
                    self.config.retrain.min_scale))
                log.debug('decrease threholds {}, decreased results {}'.format(
                    self.target_layer,
                    self._fetch_scale()
                ))
            else:
                for o in self.nets[0].overriders:
                    if o.name == self.target_layer:
                        o._scale_roll_back()
                self.cont[self.target_layer] = False

            # trace back the ckpt
            self.save_checkpoint(self.best_ckpt)
            # fetch a new layer to retrain
            self.profile_overrider()
            self.overriders_refresh()
            self.reset_num_epochs()
            return True

    def _decrease_scale(self):
        # decrease scale factor, for quantizer, this factor might be 1
        factor = self.config.retrain.scale_update_factor
        for o in self.nets[0].overriders:
            if o.name == self.target_layer:
                o._scale_roll_back()
                o._scale_update(factor)
                record = o.scale
        log.debug('decrease scaling factor to {}'.format(record))

    def log_thresholds(self, loss, acc):
        _, _, prev_loss = self.log.get(self.target_layer, [None, None, None])
        for o in self.nets[0].overriders:
            if o.name == self.target_layer:
                value = getattr(o, self.threshold_name)
                break
        if prev_loss is None:
            self.log[self.target_layer] = (value, loss, acc)
        else:
            if acc > self.acc_base:
                self.log[self.target_layer] = (value, loss, acc)
