import math
import sys
import pdb
import numpy as np
import tensorflow as tf

from mayo.log import log
from mayo.train import Train


class Retrain_Base(Train):
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
                    self.checkpoint.save('latest')

    def _init_retrain(self):
        self._reset_stats()
        self._reset_vars()
        for o in self.nets[0].overriders:
            o._setup(self)
        self.threshold_name = str(self.config.retrain.name)
        self.profile_overrider(self.threshold_name, 'scale', start=True)
        self.profile_for_one_epoch()
        self._reset_stats()
        self.overriders_refresh()

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

            if epoch >= iter_max_epoch and epoch > 0:
                self.retrain_cnt += 1
                self.reset_num_epochs()
                self.log_thresholds(self.loss_avg, self.acc_avg)
                # all layers done
                return self.update_policy()
        return True

    def update_policy(self):
        raise NotImplementedError(
            'Method of update policy is not implemented.')

    def forward_policy(self, floor_epoch):
        raise NotImplementedError(
            'Method of forward policy is not implemented.')

    def log_thresholds(self, loss, acc):
        raise NotImplementedError(
            'Method of logging threholds is not implemented.')

    def _fetch_scale(self):
        for o in self.nets[0].overriders:
            if o.name == self.target_layer:
                return o.scale

    def profile_overrider(self, threshold_name, scale_name, start=False):
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
            # d[name] = self.run(o.after).size
            d[name] = self._metric_clac(o)
            thresholds[name] = getattr(o, threshold_name)
            scales[name] = getattr(o, scale_name)
        check_bias = self.config.retrain.bias
        for key in sorted(d, key=d.get):
            log.debug('key is {} cont is {}'.format(key, self.cont[key]))
            if check_bias:
                if self.cont[key]:
                    self.priority_list.append(key)
            else:
                if self.cont[key] and ('biases' not in key):
                    self.priority_list.append(key)
        log.debug('display thresholds')
        log.debug('{}'.format(thresholds))
        log.debug('display scales')
        log.debug('{}'.format(scales))
        # log.debug('display profiling info')
        # log.debug('{}'.format(d))
        log.debug('display priority list info')
        log.debug('{}'.format(self.priority_list))
        if self.priority_list == []:
            log.debug('list is empty!!')
        else:
            self.target_layer = self.priority_list.pop()

    def empty_eval_run(self):
        tasks = []
        tasks.append(tf.assign_add(
            self.imgs_seen, self.config.system.batch_size))
        tasks.append(self.loss)
        tasks.append(self.accuracy)
        tasks.append(self.num_epochs)
        return tasks

    def profile_for_one_epoch(self):
        log.info('Start profiling for one epoch')
        epoch = 0
        self._reset_stats()
        self.reset_num_epochs()
        tolerance = self.config.retrain.tolerance
        while epoch < 1.0:
            _, loss, acc, epoch = self.run(
                self.empty_eval_run())
            self.loss_total += loss
            self.acc_total += acc
            self.step += 1
        self.loss_base = self.loss_total / float(self.step) * (1 + tolerance)
        self.acc_base = self.acc_total / float(self.step) * (1 - tolerance)
        self._reset_stats()
        self.reset_num_epochs()
        log.debug('profiled baselines, loss is {}, acc is {}'.format(
            self.loss_base,
            self.acc_base,
        ))

    def _metric_clac(self, o):
        metric_value = num_elements = self.run(o.after).size
        if hasattr(o, '_mask'):
            valid_elements = np.count_nonzero(self.run(o._mask))
            density = valid_elements / float(num_elements)
            metric_value *= density
        if hasattr(o, 'width'):
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


class Retrain_global(Retrain_Base):
    def overriders_refresh(self):
        check_bias = self.config.retrain.bias
        if check_bias:
            for o in self.nets[0].overriders:
                o._threshold_update()
                o.should_update = True
        else:
            for o in self.nets[0].overriders:
                if 'biases' not in o.name:
                    o._threshold_update()
                    o.should_update = True
        self.overriders_update()

    def forward_policy(self, floor_epoch):
        log.debug('Targeting on {}'.format(self.target_layer))
        log.debug('log: {}'.format(self.log))
        with log.demote():
            self.checkpoint.save(
                'th-' + str(self.retrain_cnt) + '-' + str(floor_epoch))
        self.best_ckpt = 'th-' + str(self.retrain_cnt) + '-' \
            + str(floor_epoch)
        self._cp_epoch = floor_epoch
        self.retrain_cnt += 1
        self.log_thresholds(self.loss_avg, self.acc_avg)
        self.profile_overrider(self.threshold_name, 'scale')
        self.overriders_refresh()
        self.reset_num_epochs()
        return True

    def log_thresholds(self, loss, acc):
        _, _, prev_loss = self.log.get(self.target_layer, [None, None, None])
        for o in self.nets[0].overriders:
            value = getattr(o, self.threshold_name)
            if prev_loss is None:
                self.log[o.name] = (value, loss, acc)
            else:
                if acc > self.acc_base:
                    self.log[self.target_layer] = (value, loss, acc)

    def update_policy(self):
        if self._fetch_scale() >= self.config.retrain.min_scale:
            self._decrease_scale()
            log.debug('min scale is {}'.format(self.config.retrain.min_scale))
            log.debug('decrease threholds to {}'.format(
                self._fetch_scale()
            ))
            return True
        else:
            for o in self.nets[0].overriders:
                self.cont[self.target_layer] = False
            log.debug('all layers done')
            return False

    def _decrease_scale(self):
        # decrease scale factor, for quantizer, this factor might be 1
        check_bias = self.config.retrain.bias
        factor = self.config.retrain.scale_update_factor
        if check_bias:
            for o in self.nets[0].overriders:
                o._scale_roll_back()
                o._scale_update(factor)
                record = o.scale
        else:
            for o in self.nets[0].overriders:
                if 'biases' not in o.name:
                    o._scale_roll_back()
                    o._scale_update(factor)
                    record = o.scale
        log.debug('decrease scaling factor to {}'.format(record))


class Retrain_layerwise(Retrain_Base):
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
            self.checkpoint.save(
                'th-' + str(self.retrain_cnt) + '-' + str(floor_epoch))
        self.best_ckpt = 'th-' + str(self.retrain_cnt) + '-' \
            + str(floor_epoch)
        self._cp_epoch = floor_epoch
        self.retrain_cnt += 1
        self.log_thresholds(self.loss_avg, self.acc_avg)
        self.profile_overrider(self.threshold_name, 'scale')
        self.overriders_refresh()
        self.reset_num_epochs()
        return True

    def update_policy(self):
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
            self._control_threholds()
            # trace back the ckpt
            self.checkpoint.load(self.best_ckpt)
            # fetch a new layer to retrain
            self.profile_overrider(self.threshold_name, 'scale')
            self.overriders_refresh()
            self.reset_num_epochs()
            return True

    def _control_threholds(self):
        if self._fetch_scale() >= self.config.retrain.min_scale:
            self._decrease_scale()
            log.debug('min scale is {}'.format(self.config.retrain.min_scale))
            log.debug('decrease threholds at {}, decreased result is {}'.format(
                self.target_layer,
                self._fetch_scale()
            ))
        else:
            for o in self.nets[0].overriders:
                if o.name == self.target_layer:
                    o._scale_roll_back()
            self.cont[self.target_layer] = False

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
