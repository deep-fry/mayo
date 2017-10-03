import math

from mayo.log import log
from mayo.train import Train


class Retrain(Train):
    def retrain(self):
        log.debug('Retraining start.')
        try:
            # train iterations
            self.log = {}
            self.loss_total = 0
            self.acc_total = 0
            self.step = 0
            self.target_layer = None
            self.loss_avg = None
            self.prune_cnt = 0
            self.best_ckpt = None
            self._profile_pruner()
            self._profile_loss()
            self._increment_c_rate()
            self.overriders_update()
            while self._retrain_iteration():
                pass
        except KeyboardInterrupt:
            log.info('Stopped.')
            save = self.config.system.checkpoint.get('save', {})
            if save:
                countdown = save.get('countdown', 0)
                if log.countdown('Saving checkpoint', countdown):
                    self.checkpoint.save('latest')

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
                self._log_thresholds(self.loss_avg, self.acc_avg)
                self._update_progress(epoch, loss, acc, 'saving')
                with log.demote():
                    self.checkpoint.save(
                        'prune-' + str(self.prune_cnt) + '-' + str(floor_epoch))
                self.best_ckpt = 'prune-' + str(self.prune_cnt) + '-' \
                    + str(floor_epoch)
                self._cp_epoch = floor_epoch
                self.prune_cnt += 1
                self._increment_c_rate()
                self.overriders_update()
                self.reset_num_epochs()
                return True

            iter_max_epoch = self.config.model.layers.iter_max_epoch
            if epoch >= iter_max_epoch and epoch > 0:
                self.prune_cnt += 1
                self.reset_num_epochs()
                self._log_thresholds(self.loss_avg, self.acc_avg)
                print('log: {}'.format(self.log))
                # all layers done
                if self.priority_list == []:
                    print('pruning done')
                    return False
                else:
                    # current layer is done
                    # trace back the ckpt
                    self.checkpoint.load(self.best_ckpt)
                    # fetch a new layer to retrain
                    self.target_layer = self.priority_list.pop()
                    self._control_updates()
                    self._increment_c_rate()
                    self.overriders_update()
        return True

    def _log_thresholds(self, loss, acc):
        _, _, prev_loss = self.log.get(self.target_layer, [None, None, None])
        for n in self.nets:
            for o in n.overriders:
                if o._mask.name == self.target_layer:
                    value = o.alpha
                    break
        if prev_loss is None:
            self.log[self.target_layer] = (value, loss, acc)
            return True
        else:
            if loss > self.loss_baseline:
                return False
            else:
                self.log[self.target_layer] = (value, loss, acc)
                return True

    def _increment_c_rate(self):
        for n in self.nets:
            for o in n.overriders:
                if o._mask.name == self.target_layer:
                    o._threshold_update(self)

    def _control_updates(self):
        for n in self.nets:
            for o in n.overriders:
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
        log.info('profiled baselines, acc is {}, loss is {}'.format(
            self.loss_baseline,
            self.acc_base,
        ))

    def _profile_pruner(self):
        self.best_ckp = 'prtrained'
        self.priority_list = []
        d = {}
        self.layerwise_losses = {}
        for o in self.nets[0].overriders:
            name = o._mask.name
            d[name] = o._mask.shape.num_elements()
            self.layerwise_losses[name] = []
        for key in sorted(d, key=d.get):
            self.priority_list.append(key)
        for n in self.nets:
            for o in n.overriders:
                o.should_update = False
        self.target_layer = self.priority_list.pop()
