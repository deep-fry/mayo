import re
import os
import time

import yaml
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from mayo.net import Net
from mayo.util import memoize, import_from_dot_path
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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._graph = tf.Graph()
        self._preprocessor = Preprocess(self.config)

    @property
    @memoize
    def global_step(self):
        initializer = tf.constant_initializer(0)
        global_step = tf.get_variable(
            'global_step', [], initializer=initializer, trainable=False)
        return global_step

    @property
    @memoize
    def learning_rate(self):
        learn_params = self.config.train.learning_rate
        rate = learn_params.initial_learning_rate
        step = self.global_step
        batches_per_epoch = self.config.dataset.num_examples_per_epoch.train
        batches_per_epoch /= self.config.train.batch_size
        decay_steps = int(
            batches_per_epoch * learn_params.num_epochs_per_decay)
        decay_factor = learn_params.learning_rate_decay_factor
        return tf.train.exponential_decay(
            rate, step, decay_steps, decay_factor, staircase=True)

    @property
    @memoize
    def optimizer(self):
        params = self.config.train.optimizer
        optimizer_class = import_from_dot_path(params.pop('type'))
        return optimizer_class(self.learning_rate, **params)

    def tower_loss(self, images, labels, reuse):
        self._net = Net(
            self.config, images, labels, graph=self._graph, reuse=reuse)
        return self._net.loss()

    def _setup_gradients(self):
        config = self.config.train
        # ensure batch size is divisible by number of gpus
        if config.batch_size % config.num_gpus != 0:
            raise ValueError('Batch size must be divisible by number of GPUs')
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
                cpu_ctx = slim.arg_scope(
                    [slim.model_variable, slim.variable], device='/cpu:0')
                with cpu_ctx:
                    # loss from the final tower
                    self._loss = self.tower_loss(
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
        summaries.append(
            tf.summary.scalar('learning_rate', self.learning_rate))
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

    def _update_progress(self, step, loss_val):
        duration = time.time() - self._step_time
        imgs_per_sec = self.config.train.batch_size / float(duration)
        info = 'step {}, loss = {:.3f} '.format(step, loss_val)
        info += '({:.1f} imgs/sec; {:.3f} sec/batch)'.format(
            imgs_per_sec, duration)
        print(info)

    @property
    @memoize
    def _summary_writer(self):
        return tf.summary.FileWriter('summaries/', graph=self._graph)

    def _save_summary(self, step):
        print('Saving summaries...')
        summary = self._session.run(self._summary_op)
        self._summary_writer.add_summary(summary, step)

    def _train(self):
        print('Instantiating...')
        self._setup_gradients()
        self._setup_train_operation()
        self._init_session()
        checkpoint = CheckpointHandler(
            self._session, self.config.name, self.config.dataset.name)
        step = checkpoint.load()
        tf.train.start_queue_runners(sess=self._session)
        self._net.save_graph()
        print('Training start')
        # train iterations
        config = self.config.train
        try:
            while step < config.max_steps:
                self._step_time = time.time()
                _, loss_val = self._session.run([self._train_op, self._loss])
                if np.isnan(loss_val):
                    raise ValueError('Model diverged with loss = NaN')
                if step % 10 == 0:
                    self._update_progress(step, loss_val)
                if step % 100 == 0:
                    self._save_summary(step)
                step += 1
                if step % 5000 == 0 or step == config.max_steps:
                    checkpoint.save(step)
        except KeyboardInterrupt:
            print('Stopped, saving checkpoint in 3 seconds.')
            time.sleep(3)
            checkpoint.save(step)

    def train(self):
        with self._graph.as_default(), tf.device('/cpu:0'):
            self._train()
