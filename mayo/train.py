import time
import math

import tensorflow as tf

from mayo.log import log
from mayo.util import (
    delta, every, moving_metrics, memoize_method, object_from_params, flatten)
from mayo.session import Session


class Train(Session):
    mode = 'train'

    def __init__(self, config):
        super().__init__(config)
        self._cp_epoch = ''
        with self.as_default():
            self._init()

    @property
    @memoize_method
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

    @property
    @memoize_method
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

    def _gradient_iterator(self, net):
        loss = net.loss()
        net_acc = net.accuracy()
        accuracy = tf.reduce_sum(tf.cast(net_acc, tf.float32))
        accuracy /= net_acc.shape.num_elements()
        grads = self.optimizer.compute_gradients(loss)
        updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        return loss, accuracy, grads, updates, summaries

    def _setup_gradients(self):
        log.debug('Initializing gradients...')
        losses, accuracies, tower_grads, updates, summaries = zip(
            *self.net_map(self._gradient_iterator))
        self._gradients = self._average_gradients(tower_grads)
        # update ops
        self._update_ops = flatten(updates)
        self._imgs_seen_op = tf.assign_add(
            self.imgs_seen, self.config.system.batch_size)
        # summaries
        self._loss = tf.reduce_sum(losses)
        self._acc = tf.reduce_mean(accuracies)
        summaries += (
            tf.summary.scalar('learning_rate', self.learning_rate),
            tf.summary.scalar('loss', self._loss))
        self._summary_op = tf.summary.merge(summaries)

    def _setup_train_operation(self):
        log.debug('Initializing training operations...')
        app_grad_op = self.optimizer.apply_gradients(self._gradients)
        ops = [app_grad_op]
        ma_op = self.moving_average_op()
        if ma_op:
            ops.append(ma_op)
        ops += self._update_ops
        log.debug('Train operations: {}'.format(ops))
        self._train_op = tf.group(*ops)

    def _init(self):
        self._setup_gradients()
        self._setup_train_operation()
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
        info = 'epoch: {:.2f} | loss: {:10f}{:5} | acc: {:5.2f}%'
        if cp_epoch:
            info += ' | ckpt: {}'
        loss_mean, loss_std = moving_metrics(
            'train.loss', loss, over=metric_count)
        loss_std = '±{}%'.format(int(loss_std / loss_mean * 100))
        acc_mean = moving_metrics(
            'train.accuracy', accuracy * 100, std=False, over=metric_count)
        info = info.format(epoch, loss_mean, loss_std, acc_mean, cp_epoch)
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

    @property
    @memoize_method
    def _summary_writer(self):
        path = self.config.system.search_path.summary[0]
        return tf.summary.FileWriter(path, graph=self.graph)

    def _save_summary(self, epoch):
        summary = self.run(self._summary_op)
        self._summary_writer.add_summary(summary, epoch)

    def once(self):
        tasks = [
            self._train_op, self._imgs_seen_op,
            self._loss, self._acc, self.num_epochs]
        noop, imgs_seen, loss, acc, num_epochs = self.run(tasks)
        return loss, acc, num_epochs

    def update_overriders(self):
        log.info('Updating overrider variables...')
        for n in self.nets:
            for o in n.overriders:
                o.update(self)

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
