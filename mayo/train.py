import time
import math

import tensorflow as tf

from mayo.log import log
from mayo.util import Percent, memoize_property, object_from_params
from mayo.session import Session


class Train(Session):
    mode = 'train'

    def __init__(self, config):
        super().__init__(config)
        with self.as_default():
            self._setup_train_operation()
            self._setup_summaries()
            self._init()
        self._cp_epoch = ''

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
        ops['imgs_seen'] = tf.assign_add(self.imgs_seen, self.batch_size)
        ops['app_grad'] = self.optimizer.apply_gradients(self._gradients)
        ma_op = self.moving_average_op()
        if ma_op:
            ops['avg'] = ma_op
        # update ops
        update_ops = list(self.get_collection(tf.GraphKeys.UPDATE_OPS))
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
        self.load_checkpoint(self.config.system.checkpoint.load)
        # final debug outputs
        if not log.is_enabled('debug'):
            return
        lr = self.run(self.learning_rate)
        log.debug('Current learning rate is {}.'.format(lr))

    def _update_progress(self, epoch, loss, accuracy, cp_epoch):
        metric_count = self.config.system.log.metrics_history_count
        if not isinstance(cp_epoch, str):
            cp_epoch = '{:.2f}'.format(cp_epoch)
        info = 'epoch: {:.2f} | loss: {:10f}{:5} | acc: {}'
        if cp_epoch:
            info += ' | ckpt: {}'
        loss_mean, loss_std = self.change.moving_metrics(
            'loss', loss, over=metric_count)
        loss_std = '±{}'.format(Percent(loss_std / loss_mean))
        acc_mean = self.change.moving_metrics(
            'accuracy', accuracy, std=False, over=metric_count)
        info = info.format(
            epoch, loss_mean, loss_std, Percent(acc_mean), cp_epoch)
        # performance
        interval = self.change.delta('step.duration', time.time())
        if interval != 0:
            imgs = epoch * self.config.dataset.num_examples_per_epoch.train
            imgs_per_step = self.change.delta('step.images', imgs)
            imgs_per_sec = imgs_per_step / float(interval)
            imgs_per_sec = self.change.moving_metrics(
                'imgs_per_sec', imgs_per_sec, std=False, over=metric_count)
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
        self.change.reset('checkpoint.epoch')

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
        summary_delta = self.change.delta('summary.epoch', epoch)
        if system.summary.save and summary_delta >= 0.1:
            self._save_summary(epoch)
        floor_epoch = math.floor(epoch)
        cp_interval = system.checkpoint.get('save.interval', 0)
        if self.change.every('checkpoint.epoch', floor_epoch, cp_interval):
            self._update_progress(epoch, loss, acc, 'saving')
            with log.demote():
                self.save_checkpoint(floor_epoch)
            self._cp_epoch = floor_epoch
        if system.max_epochs and floor_epoch >= system.max_epochs:
            log.info('Maximum epoch count reached.')
            if self._cp_epoch and floor_epoch > self._cp_epoch:
                log.info('Saving final checkpoint...')
                self.save_checkpoint(floor_epoch)
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
                    self.save_checkpoint('latest')
