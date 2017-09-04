import os
import time

import numpy as np
import tensorflow as tf

from mayo.log import log
from mayo.net import Net
from mayo.util import memoize, object_from_params
from mayo.preprocess import Preprocess
from mayo.checkpoint import CheckpointHandler


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


class Train(object):
    average_count = 100

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._graph = tf.Graph()
        self._nets = []
        self._preprocessor = Preprocess(self.config)

    @property
    @memoize
    def global_step(self):
        initializer = tf.constant_initializer(0)
        global_step = tf.get_variable(
            'global_step', [], initializer=initializer, trainable=False,
            dtype=tf.int32)
        return global_step

    @property
    @memoize
    def learning_rate(self):
        params = self.config.train.learning_rate
        lr_class, params = object_from_params(params)
        if lr_class is tf.train.piecewise_constant:
            step_name = 'x'
        else:
            step_name = 'global_step'
        params[step_name] = self.global_step
        return lr_class(**params)

    @property
    @memoize
    def optimizer(self):
        params = self.config.train.optimizer
        optimizer_class, params = object_from_params(params)
        return optimizer_class(self.learning_rate, **params)

    def tower_loss(self, images, labels, reuse):
        net = Net(
            self.config, images, labels, True, graph=self._graph, reuse=reuse)
        self._nets.append(net)
        return net.loss(), net.accuracy()

    def _setup_gradients(self):
        config = self.config.system
        # ensure batch size is divisible by number of gpus
        if config.batch_size % config.num_gpus != 0:
            raise ValueError(
                'Batch size must be divisible by number of devices')
        # initialize images and labels
        images_splits, labels_splits = self._preprocessor.preprocess_train()
        # for each gpu
        iterator = enumerate(zip(images_splits, labels_splits))
        tower_grads = []
        reuse = None
        for i, (images_split, label_split) in iterator:
            # loss with the proper nested contexts
            name = 'tower_{}'.format(i)
            with tf.device('/gpu:{}'.format(i)), tf.name_scope(name):
                # loss from the final tower
                self._loss, self._acc = self.tower_loss(
                    images_split, label_split, reuse)
                reuse = True
                # batch norm updates from the final tower
                with self._graph.as_default():
                    # summaries from the final tower
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
                    self._batch_norm_updates = tf.get_collection(
                        tf.GraphKeys.UPDATE_OPS, name)
                # gradients from all towers
                grads = self.optimizer.compute_gradients(self._loss)
                tower_grads.append(grads)
        self._gradients = _average_gradients(tower_grads)
        # summaries
        summaries += [
            tf.summary.scalar('learning_rate', self.learning_rate),
            tf.summary.scalar('loss', self._loss)]
        self._summary_op = tf.summary.merge(summaries)

    def _setup_train_operation(self):
        app_grad_op = self.optimizer.apply_gradients(
            self._gradients, global_step=self.global_step)
        var_avgs = tf.train.ExponentialMovingAverage(
            self.config.train.moving_average_decay, self.global_step)
        var_avgs_op = var_avgs.apply(
            tf.trainable_variables() + tf.moving_average_variables())
        bn_op = tf.group(*self._batch_norm_updates)
        self._train_op = tf.group(app_grad_op, var_avgs_op, bn_op)

    def _init_session(self):
        # build an initialization operation to run below
        init = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        self._session = tf.Session(config=config)
        self._session.run(init)

    def _to_epoch(self, step):
        epoch = step * self.config.system.batch_size
        return epoch / float(self.config.dataset.num_examples_per_epoch.train)

    def _moving_average(self, name, value, std=True):
        name = '_ma_{}'.format(name)
        history = getattr(self, name, [])
        if len(history) == self.average_count:
            history.pop(0)
        history.append(value)
        setattr(self, name, history)
        mean = np.mean(history)
        if not std:
            return mean
        return mean, np.std(history)

    def _update_progress(self, step, loss, accuracy, cp_step):
        epoch = self._to_epoch(step)
        if not isinstance(cp_step, str):
            cp_step = '{:.2f}'.format(self._to_epoch(cp_step))
        info = 'epoch: {:.2f} | loss: {:10f}{:5}'
        info += ' | acc: {:5.2f}% | ckpt: {}'
        loss_mean, loss_std = self._moving_average('loss', loss)
        acc_percentage = np.sum(accuracy) / self.config.system.batch_size
        acc_percentage *= self.config.system.num_gpus * 100
        accuracy_mean, _ = self._moving_average('accuracy', acc_percentage)
        info = info.format(
            epoch, loss_mean, '±{}%'.format(int(loss_std / loss_mean * 100)),
            accuracy_mean, cp_step)
        # performance
        now = time.time()
        duration = now - getattr(self, '_prev_time', now)
        if duration != 0:
            num_steps = step - getattr(self, '_prev_step', step)
            imgs_per_sec = num_steps * self.config.system.batch_size
            imgs_per_sec /= float(duration)
            imgs_per_sec = self._moving_average(
                'imgs_per_sec', imgs_per_sec, std=False)
            info += ' | tp: {:4.0f}/s'.format(imgs_per_sec)
        log.info(info, update=True)
        self._prev_time = now
        self._prev_step = step

    @property
    @memoize
    def _summary_writer(self):
        path = self.config.system.search_paths.summaries[0]
        directory = os.path.join(
            path, self.config.name, self.config.dataset.name)
        return tf.summary.FileWriter(directory, graph=self._graph)

    def _save_summary(self, step):
        summary = self._session.run(self._summary_op)
        self._summary_writer.add_summary(summary, step)

    def _train(self):
        log.info('Instantiating...')
        self._setup_gradients()
        self._setup_train_operation()
        log.info('Initializing session...')
        self._init_session()
        # checkpoint
        system = self.config.system
        checkpoint = CheckpointHandler(
            self._session, self.config.name, self.config.dataset.name,
            system.checkpoint.load, system.checkpoint.save,
            system.search_paths.checkpoints)
        cp_step = step = checkpoint.load()
        curr_step = 0
        tf.train.start_queue_runners(sess=self._session)
        self._nets[0].save_graph()
        # training start
        log.info('Training start.')
        # train iterations
        max_steps = self.config.system.max_steps
        try:
            while step < max_steps or max_steps <= 0:
                _, loss, acc = self._session.run(
                    [self._train_op, self._loss, self._acc])
                if np.isnan(loss):
                    raise ValueError('Model diverged with a nan-valued loss.')
                self._update_progress(step, loss, acc, cp_step)
                if curr_step % 1000 == 0:
                    self._save_summary(step)
                curr_step += 1
                if curr_step % 5000 == 0 or curr_step == max_steps:
                    if self.config.system.checkpoint.save:
                        self._update_progress(step, loss, acc, 'saving')
                        with log.use_level('warn'):
                            checkpoint.save(step)
                        cp_step = step
                step += 1
        except KeyboardInterrupt:
            log.info('Stopped, saving checkpoint in 3 seconds.')
        try:
            time.sleep(3)
        except KeyboardInterrupt:
            return
        checkpoint.save(step)

    def train(self):
        with self._graph.as_default(), tf.device('/cpu:0'):
            self._train()
