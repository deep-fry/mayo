import time
import math

import tensorflow as tf

from mayo.log import log
from mayo.util import (
    delta, every, moving_metrics, Percent,
    memoize_property, object_from_params)
from mayo.session import Session


class Train(Session):
    mode = 'train'

    def __init__(self, config):
        super().__init__(config)
        self._cp_epoch = ''
        with self.as_default():
            self._init()

    @memoize_property
    def learning_rate(self):
        params = self.config.train.learning_rate
        lr_class, params = object_from_params(params)
        if lr_class is tf.train.piecewise_constant:
            # `tf.train.piecewise_constant` uses argument name 'x' instead
            # just to make life more difficult
            step_name = 'x'
        else:
            step_name = 'global_step'
        params[step_name] = self.num_epochs
        log.debug(
            'Using learning rate {!r} with params {}.'
            .format(lr_class.__name__, params))
        with self.as_default():
            return lr_class(**params)

    @memoize_property
    def optimizer(self):
        params = self.config.train.optimizer
        optimizer_class, params = object_from_params(params)
        log.debug('Using optimizer {!r}.'.format(optimizer_class.__name__))
        with self.as_default():
            return optimizer_class(self.learning_rate, **params)

    @staticmethod
    def _average_gradients(tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                # add 0 dimension to the gradients to represent the tower
                g = tf.expand_dims(g, 0)
                grads.append(g)
            # average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
            # simply return the first tower's pointer to the Variable
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    @memoize_property
    def _gradients(self):
        grads_func = lambda net: self.optimizer.compute_gradients(net.loss())
        tower_grads = self.net_map(grads_func)
        return self._average_gradients(tower_grads)

    def _setup_train_operation(self):
        ops = {}
        ops['imgs_seen'] = tf.assign_add(
            self.imgs_seen, self.config.system.batch_size)
        ops['app_grad'] = self.optimizer.apply_gradients(self._gradients)
        ma_op = self.moving_average_op()
        if ma_op:
            ops['avg'] = ma_op
        # update ops
        update_ops = self.get_collection(tf.GraphKeys.UPDATE_OPS)
        ops['update'] = tf.group(*update_ops, name='update')
        log.debug('Using update operations: {}'.format(update_ops))
        log.debug('Using training operations: {}'.format(ops))
        self._train_op = ops

    def _setup_summaries(self):
        if not self.config.system.summary.save:
            return
        summaries = list(self.get_collection(tf.GraphKeys.SUMMARIES))
        summaries += [
            tf.summary.scalar('learning_rate', self.learning_rate),
            tf.summary.scalar('loss', self.loss)]
        self._summary_op = tf.summary.merge(summaries)

    def _init(self):
        self._setup_train_operation()
        self._setup_summaries()
        self.init()
        self.checkpoint.load(self.config.system.checkpoint.load)
        # final debug outputs
        if not log.is_enabled('debug'):
            return
        lr = self.run(self.learning_rate)
        log.debug('Learning rate is {}.'.format(lr))

    def _update_progress(self, epoch, loss, accuracy, cp_epoch):
        metric_count = self.config.system.log.metrics_history_count
        if not isinstance(cp_epoch, str):
            cp_epoch = '{:.2f}'.format(cp_epoch)
        info = 'epoch: {:.2f} | loss: {:10f}{:5} | acc: {}'
        if cp_epoch:
            info += ' | ckpt: {}'
        loss_mean, loss_std = moving_metrics(
            'train.loss', loss, over=metric_count)
        loss_std = '±{}'.format(Percent(loss_std / loss_mean))
        acc_mean = moving_metrics(
            'train.accuracy', accuracy, std=False, over=metric_count)
        info = info.format(
            epoch, loss_mean, loss_std, Percent(acc_mean), cp_epoch)
        # performance
        interval = delta('train.duration', time.time())
        if interval != 0:
            imgs = epoch * self.config.dataset.num_examples_per_epoch.train
            imgs_per_step = delta('train.step.imgs', imgs)
            imgs_per_sec = imgs_per_step / float(interval)
            imgs_per_sec = moving_metrics(
                'train.imgs_per_sec', imgs_per_sec,
                std=False, over=metric_count)
            info += ' | tp: {:4.0f}/s'.format(imgs_per_sec)
        log.info(info, update=True)

    @memoize_property
    def _summary_writer(self):
        path = self.config.system.search_path.summary[0]
        return tf.summary.FileWriter(path, graph=self.graph)

    def _save_summary(self, epoch):
        summary = self.run(self._summary_op)
        self._summary_writer.add_summary(summary, epoch)

    def reset_num_epochs(self):
        log.info('Reseting number of training epochs of the model...')
        with self.as_default():
            self.run(tf.assign(self.imgs_seen, 0))

    def once(self):
        tasks = [self._train_op, self.loss, self.accuracy, self.num_epochs]
        noop, loss, acc, num_epochs = self.run(tasks)
        return loss, acc, num_epochs

    def _overriders_call(self, func_name):
        # it is sufficient to use the first net, as overriders
        # share internal variables
        for o in self.nets[0].overriders:
            getattr(o, func_name)(self)

    def overriders_assign(self):
        log.info('Assigning overridden values of parameters to parameters...')
        self._overriders_call('assign')

    def overriders_update(self):
        log.info('Updating overrider internal variables...')
        self._overriders_call('update')

    def overriders_reset(self):
        log.info('Resetting overriders internal variables...')
        self._overriders_call('reset')

    def _iteration(self):
        system = self.config.system
        loss, acc, epoch = self.once()
        if math.isnan(loss):
            raise ValueError('Model diverged with a nan-valued loss.')
        self._update_progress(epoch, loss, acc, self._cp_epoch)
        summary_delta = delta('train.summary.epoch', epoch)
        if system.summary.save and summary_delta >= 0.1:
            self._save_summary(epoch)
        floor_epoch = math.floor(epoch)
        cp_interval = system.checkpoint.get('save.interval', 0)
        if every('train.checkpoint.epoch', floor_epoch, cp_interval):
            self._update_progress(epoch, loss, acc, 'saving')
            with log.demote():
                self.checkpoint.save(floor_epoch)
            self._cp_epoch = floor_epoch
        if system.max_epochs and floor_epoch >= system.max_epochs:
            log.info('Maximum epoch count reached.')
            if self._cp_epoch and floor_epoch > self._cp_epoch:
                log.info('Saving final checkpoint...')
                self.checkpoint.save(floor_epoch)
            return False
        return True

    def train(self):
        log.debug('Training start.')
        try:
            # train iterations
            while self._iteration():
                pass
        except KeyboardInterrupt:
            log.info('Stopped.')
            save = self.config.system.checkpoint.get('save', {})
            if save:
                countdown = save.get('countdown', 0)
                if log.countdown('Saving checkpoint', countdown):
                    self.checkpoint.save('latest')

    def retrain(self):
        log.debug('Retraining start.')
        try:
            # train iterations
            self.log = {}
            self.loss_total = 0
            self.step = 0
            self.target_layer = None
            self.loss_avg = None
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
        if every('train.checkpoint.epoch', floor_epoch, cp_interval):
            self.curr_loss_avg = self.loss_total / float(self.step)
            self.loss_total = 0
            self.step = 0
            if self.loss_avg is None or self.loss_avg > self.curr_loss_avg:
                self._update_progress(epoch, loss, acc, 'saving')
                with log.demote():
                    self.checkpoint.save(floor_epoch)
                self._cp_epoch = floor_epoch
                self.loss_avg = self.curr_loss_avg
        iter_max_epoch = self.config.model.layers.iter_max_epoch
        if epoch >= iter_max_epoch and epoch > 0:
            self.reset_num_epochs()
            if self.loss_avg is None or self.loss_avg > self.curr_loss_avg:
                self.checkpoint.save(floor_epoch)
                self.loss_avg = self.curr_loss_avg
                self._cp_epoch = floor_epoch
            print('Best loss avg {}, found at {}'.format(
                self.loss_avg,
                self._cp_epoch
            ))
            is_layer_continue = self._log_thresholds(self.loss_avg)
            if is_layer_continue:
                return True
            else:
                # all layers done
                if self.priority_list == []:
                    return False
                else:
                    # fetch a new layer to retrain
                    self.target_layer = self.priority_list.pop()
                    self._control_updates()
                    self.overriders_update()
                    return True
        else:
            return True

    def _log_thresholds(self, loss):
        tolerance = self.config.model.layers.tolerance
        if self.target_layer in self.log.keys():
            prev_value, prev_loss = self.log[self.target_layer]
        else:
            prev_value, prev_loss = (None, None)
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
        scale_interval = self.config.model.layers.scale
        iter_max_epoch = self.config.model.layers.iter_max_epoch
        for n in self.nets:
            for o in n.overriders:
                if o._mask.name == self.target_layer:
                    o._threshold_update(
                        self.tf_session, scale_interval, iter_max_epoch,
                        self.num_epochs)

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
