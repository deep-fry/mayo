import math
import sys
import pdb
import numpy as np

from mayo.log import log
from mayo.train import Train


class Retrain(Train):
    def retrain(self):
        log.debug('Retraining start.')
        try:
            self._init_prune()
            while self._retrain_iteration():
                pass
        except KeyboardInterrupt:
            log.info('Stopped.')
            save = self.config.system.checkpoint.get('save', {})
            if save:
                countdown = save.get('countdown', 0)
                if log.countdown('Saving checkpoint', countdown):
                    self.checkpoint.save('latest')

    def _init_prune(self):
        self.log = {}
        # train iterations
        self.loss_total = 0
        self.acc_total = 0
        self.step = 0
        self.target_layer = None
        self.loss_avg = None
        self.prune_cnt = 0
        self.best_ckpt = None
        for o in self.nets[0].overriders:
            o._set_up_scale(self)
        # profile
        self._profile_pruner(start=True)
        self._profile_loss()
        self._increment_c_rate()
        self.overriders_update()
        self._control_updates()

    def _retrain_iteration(self):
        system = self.config.system
        loss, acc, epoch = self.once()

        self.loss_total += loss
        self.acc_total += acc
        self.step += 1

        if math.isnan(loss):
            raise ValueError('Model diverged with a nan-valued loss.')
        self._update_progress(epoch, loss, acc, self._cp_epoch)
        summary_delta = self.change.delta('summary.epoch', epoch)

        if system.summary.save and summary_delta >= 0.1:
            self._save_summary(epoch)
        floor_epoch = math.floor(epoch)
        cp_interval = system.checkpoint.get('save.interval', 0)

        if self.change.every('checkpoint.epoch', floor_epoch, cp_interval):
            self.loss_avg = self.loss_total / float(self.step)
            self.acc_avg = self.acc_total / float(self.step)
            self.loss_total = 0
            self.acc_total = 0
            self.step = 0

            # if self.loss_avg <= self.loss_baseline:
            if self.acc_avg >= self.acc_base:
                log.debug('Increase c rate on {}'.format(self.target_layer))
                log.debug('log: {}'.format(self.log))
                # self._update_progress(epoch, loss, acc, 'saving')
                with log.demote():
                    self.checkpoint.save(
                        'prune-' + str(self.prune_cnt) + '-' + str(floor_epoch))
                self.best_ckpt = 'prune-' + str(self.prune_cnt) + '-' \
                    + str(floor_epoch)
                self._cp_epoch = floor_epoch
                self.prune_cnt += 1
                self._log_thresholds(self.loss_avg, self.acc_avg)
                self._profile_pruner()
                self._control_updates()
                self._increment_c_rate()
                self.overriders_update()
                self.reset_num_epochs()
                return True

            iter_max_epoch = self.config.model.layers.iter_max_epoch
            if epoch >= iter_max_epoch and epoch > 0:
                self.prune_cnt += 1
                self.reset_num_epochs()
                self._log_thresholds(self.loss_avg, self.acc_avg)
                # all layers done
                finished = self.cont[self.target_layer] is False
                if self.priority_list == [] and finished:
                    log.info('pruning done, model stored at {}'.format(
                        self.best_ckpt))
                    return False
                else:
                    # current layer is done
                    self._control_c_rates()
                    # trace back the ckpt
                    self.checkpoint.load(self.best_ckpt)
                    # fetch a new layer to retrain
                    self._profile_pruner()
                    self._control_updates()
                    self._increment_c_rate()
                    self.overriders_update()
                    self.reset_num_epochs()
        return True

    def _control_c_rates(self):
        if self._fetch_scale() >= self.config.model.layers.min_scale:
            self._decrease_scale()
        else:
            self.cont[self.target_layer] = False

    def _fetch_scale(self):
        for o in self.nets[0].overriders:
            if o._mask.name == self.target_layer:
                return o.scale

    def _log_thresholds(self, loss, acc):
        _, _, prev_loss = self.log.get(self.target_layer, [None, None, None])
        for o in self.nets[0].overriders:
            if o._mask.name == self.target_layer:
                value = o.alpha
                break
        if prev_loss is None:
            self.log[self.target_layer] = (value, loss, acc)
        else:
            if acc > self.acc_base:
                self.log[self.target_layer] = (value, loss, acc)

    def _decrease_scale(self):
        for o in self.nets[0].overriders:
            if o._mask.name == self.target_layer:
                o._scale_update(0.5)
                record = o.scale
        log.debug('decrease scaling factor to {}'.format(record))

    def _increment_c_rate(self):
        for o in self.nets[0].overriders:
            if o._mask.name == self.target_layer:
                o._threshold_update()

    def _control_updates(self):
        for o in self.nets[0].overriders:
            if o._mask.name == self.target_layer:
                o.should_update = True

    def _profile_loss(self):
        log.info('Start profiling for one epoch')
        step = 0
        loss_total = 0
        acc_total = 0
        epoch = 0
        self.reset_num_epochs()
        tolerance = self.config.model.layers.tolerance
        while epoch < 1.0:
            loss, acc, epoch = self.once()
            loss_total += loss
            acc_total += acc
            step += 1
        self.loss_baseline = loss_total / float(step) * (1 + tolerance)
        self.acc_base = acc_total / float(step) * (1 - tolerance)
        self.reset_num_epochs()
        log.debug('profiled baselines, loss is {}, acc is {}'.format(
            self.loss_baseline,
            self.acc_base,
        ))

    def _profile_pruner(self, start=False):
        self.priority_list = []
        if start:
            self.best_ckp = 'prtrained'
            self.cont = {}
            for o in self.nets[0].overriders:
                name = o._mask.name
                self.cont[name] = True
                o.should_update = False
        d = {}
        cRates = {}
        cRates_scale = {}
        for o in self.nets[0].overriders:
            name = o._mask.name
            d[name] = np.count_nonzero(self.run(o._mask))
            cRates[name] = o.alpha
            cRates_scale[name] = o.scale
        for key in sorted(d, key=d.get):
            log.debug('key is {} cont is {}'.format(key, self.cont[key]))
            if self.cont[key] and ('biases' not in key):
                self.priority_list.append(key)
        log.debug('display cRates')
        log.debug('{}'.format(cRates))
        log.debug('display cRates scale')
        log.debug('{}'.format(cRates_scale))
        log.debug('display profiling info')
        log.debug('{}'.format(d))
        log.debug('display priority list info')
        log.debug('{}'.format(self.priority_list))
        # log.debug('display cont info')
        # log.debug('{}'.format(self.cont))
        self.target_layer = self.priority_list.pop()
