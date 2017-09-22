import time
import math

import tensorflow as tf

from mayo.log import log
from mayo.net import Net
from mayo.util import (
    delta, every, moving_metrics, memoize_method, object_from_params)
from mayo.session import Session


class Train(Session):
    def __init__(self, config):
        super().__init__(config)
        self._nets = []
        with self.as_default():
            self._init()

    @property
    @memoize_method
    def learning_rate(self):
        params = self.config.train.learning_rate
        lr_class, params = object_from_params(params)
        if lr_class is tf.train.piecewise_constant:
            # tf.train.piecewise_constant uses argument name 'x' instead of
            # 'global_step' just to make life more difficult
            step_name = 'x'
        else:
            step_name = 'global_step'
        params[step_name] = self.global_step
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

    def _tower_loss(self, images, labels, reuse):
        net = Net(self.config, images, labels, True, reuse=reuse)
        self._nets.append(net)
        return net.loss(), net.accuracy()

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

    def _setup_gradients(self):
        config = self.config.system
        # ensure batch size is divisible by number of gpus
        if config.batch_size % config.num_gpus != 0:
            raise ValueError(
                'Batch size must be divisible by number of devices')
        # initialize images and labels
        images_splits, labels_splits = self.preprocess('train')
        # for each gpu
        iterator = enumerate(zip(images_splits, labels_splits))
        tower_grads = []
        reuse = None
        for i, (images_split, label_split) in iterator:
            log.debug('Instantiating loss for GPU #{}.'.format(i))
            # loss with the proper nested contexts
            name = 'tower_{}'.format(i)
            with tf.device('/gpu:{}'.format(i)), tf.name_scope(name):
                # loss from the final tower
                self._loss, self._acc = self._tower_loss(
                    images_split, label_split, reuse)
                reuse = True
                # batch norm updates from the final tower
                # summaries from the final tower
                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
                self._batch_norm_updates = tf.get_collection(
                    tf.GraphKeys.UPDATE_OPS, name)
                # gradients from all towers
                grads = self.optimizer.compute_gradients(self._loss)
                tower_grads.append(grads)
        self._gradients = self._average_gradients(tower_grads)
        # update num imgs
        self._imgs_seen_op = tf.assign_add(self.imgs_seen, config.batch_size)
        # summaries
        summaries += [
            tf.summary.scalar('learning_rate', self.learning_rate),
            tf.summary.scalar('loss', self._loss)]
        self._summary_op = tf.summary.merge(summaries)

    def _setup_train_operation(self):
        app_grad_op = self.optimizer.apply_gradients(
            self._gradients, global_step=self.global_step)
        ops = [app_grad_op]
        avg_op = self.moving_average_op()
        if avg_op:
            ops.append(avg_op)
        bn_op = tf.group(*self._batch_norm_updates)
        ops.append(bn_op)
        self._train_op = tf.group(*ops)

    def _init(self):
        log.debug('Instantiating...')
        self._setup_gradients()
        self._setup_train_operation()
        self.init()
        self.checkpoint.load()

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
        acc_percent = sum(accuracy) / self.config.system.batch_size
        acc_percent *= self.config.system.num_gpus * 100
        acc_mean = moving_metrics(
            'train.accuracy', acc_percent, std=False, over=metric_count)
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
            self._train_op, self._loss, self._acc, self._imgs_seen_op]
        _, loss, acc, imgs_seen = self.run(tasks)
        return loss, acc, imgs_seen

    def update_overriders(self):
        log.info('Updating overrider variables...')
        for n in self._nets:
            for o in n.overriders:
                o.update(self)

    def train(self):
        imgs_per_epoch = self.config.dataset.num_examples_per_epoch.train
        log.debug('Training start.')
        cp_epoch = ''
        # train iterations
        system = self.config.system
        cp_interval = system.checkpoint.get('save.interval', 0)
        try:
            while True:
                loss, acc, imgs_seen = self.once()
                epoch = imgs_seen / imgs_per_epoch
                if math.isnan(loss):
                    raise ValueError('Model diverged with a nan-valued loss.')
                self._update_progress(epoch, loss, acc, cp_epoch)
                summary_delta = delta('train.summary.epoch', epoch)
                if system.summary.save and summary_delta >= 0.1:
                    self._save_summary(epoch)
                floor_epoch = math.floor(epoch)
                if every('train.checkpoint.epoch', floor_epoch, cp_interval):
                    self._update_progress(epoch, loss, acc, 'saving')
                    with log.demote():
                        self.checkpoint.save(floor_epoch)
                    cp_epoch = floor_epoch
                if system.max_epochs and floor_epoch >= system.max_epochs:
                    log.info('Maximum epoch count reached.')
                    if cp_epoch and floor_epoch > cp_epoch:
                        log.info('Saving final checkpoint...')
                        self.checkpoint.save(floor_epoch)
                    return
        except KeyboardInterrupt:
            log.info('Stopped.')
            save = self.config.system.checkpoint.get('save', {})
            if save:
                countdown = save.get('countdown', 0)
                if log.countdown('Saving checkpoint', countdown):
                    self.checkpoint.save('latest')
