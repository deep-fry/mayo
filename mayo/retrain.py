import math

from mayo.util import delta, every, retrain_every
from mayo.log import log
from mayo.train import Train


class Retrain(Train):
    def retrain(self):
        log.debug('Retraining start.')
        try:
            # train iterations
            self.log = {}
            self.loss_total = 0
            self.step = 0
            self.target_layer = None
            self.loss_avg = None
            self.prune_cnt = 0
            self.best_ckpt = None
            self._profile_pruner()
            self.reset_num_epochs()
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
        self._increment_c_rate()
        loss, acc, epoch = self.once()

        self.loss_total += loss
        self.step += 1

        if math.isnan(loss):
            raise ValueError('Model diverged with a nan-valued loss.')
        self._update_progress(epoch, loss, acc, self._cp_epoch)
        summary_delta = delta('train.summary.epoch', epoch)
        if system.summary.save and summary_delta >= 0.1:
            self._save_summary(epoch)
        floor_epoch = math.floor(epoch)
        cp_interval = system.checkpoint.get('save.interval', 0)
        if retrain_every('train.checkpoint.epoch', floor_epoch, cp_interval):
            self.curr_loss_avg = self.loss_total / float(self.step)
            self.loss_total = 0
            self.step = 0
            if self.loss_avg is None or self.loss_avg > self.curr_loss_avg:
                self._update_progress(epoch, loss, acc, 'saving')
                with log.demote():
                    self.checkpoint.save(
                        'prune' + str(self.prune_cnt) + '-' + str(floor_epoch))
                self.best_ckpt = 'prune' + str(self.prune_cnt) + '-' \
                    + str(floor_epoch)
                self._cp_epoch = floor_epoch
                self.loss_avg = self.curr_loss_avg
        iter_max_epoch = self.config.model.layers.iter_max_epoch
        if epoch >= iter_max_epoch and epoch > 0:
            if self.loss_avg is None or self.loss_avg > self.curr_loss_avg:
                self.checkpoint.save(
                    'prune' + str(self.prune_cnt) + '-' + str(floor_epoch))
                self.loss_avg = self.curr_loss_avg
                self._cp_epoch = floor_epoch
                self.best_ckpt = 'prune' + str(self.prune_cnt) + '-' \
                    + str(floor_epoch)
            print('Best loss avg {}, found at {}'.format(
                self.loss_avg,
                self._cp_epoch
            ))
            self.prune_cnt += 1
            self.reset_num_epochs()
            self.loss_total = 0
            self.step = 0
            is_layer_continue = self._log_thresholds(self.loss_avg)
            self.loss_avg = None
            if is_layer_continue:
                return True
            else:
                # all layers done
                if self.priority_list == []:
                    return False
                else:
                    # current layer is done
                    # trace back the ckpt
                    self.checkpoint.load(self.best_ckpt)
                    # fetch a new layer to retrain
                    self.target_layer = self.priority_list.pop()
                    self._control_updates()
                    self.overriders_update()
                    return True
        else:
            return True

    def _log_thresholds(self, loss):
        tolerance = self.config.model.layers.tolerance
        _, prev_loss = self.log.get(self.target_layer, [None, None])
        for n in self.nets:
            for o in n.overriders:
                if o._mask.name == self.target_layer:
                    value = o.alpha
                    break
        if prev_loss is None:
            self.log[self.target_layer] = (value, loss)
            return True
        else:
            if loss > (1 + tolerance) * prev_loss:
                return False
            else:
                self.log[self.target_layer] = (value, loss)
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

    def _check_loss(self):
        tolerance = self.config.model.layers.tolerance
        if self.layerwise_losses[self.target_layer] == []:
            baseline = 10000
        else:
            baseline = self.layerwise_losses[self.target_layer][-1]
        if self.loss_avg > baseline * (1 + tolerance):
            self.checkpoint.load(self.layerwise_epochs[self.target_layer][-1])
            check = False
        else:
            self.layerwise_losses[self.target_layer].append(self.loss_avg)
            check = True
            # self.layerwise_epochs[self.target_layer].append(self._cp_epoch)
        self.loss_avg = None
        self.loss_total = 0
        self.step = 0
        return check

    def _profile_pruner(self):
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
