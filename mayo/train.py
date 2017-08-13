import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from mayo.net import Net
from mayo.util import memoize, import_from_dot_path


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
        self._gradients = None
        self._batch_norm_updates = None
        self._train_op = None
        self._session = None
        self._net = None
        self._loss = None
        self._step_time = None

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
        dataset_params = self.config.dataset
        learn_params = self.config.train.learning_rate
        rate = learn_params.initial_learning_rate
        step = self.global_step
        batches_per_epoch = dataset_params.num_examples_per_epoch.train
        batches_per_epoch /= dataset_params.batch_size
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

    @memoize
    def preprocess(self):
        images, labels = self._preprocess()
        split = lambda t: tf.split(
            axis=0, num_or_size_splits=self.config.train.num_gpus, value=t)
        return split(images), split(labels)

    @memoize
    def dataset(self):
        dataset = Imagenet

    def _preprocess(self, dataset):
        raise NotImplementedError

    def tower_loss(self, images, labels, reuse):
        batch_size = images.get_shape().as_list()[0]
        self._net = Net(
            self.config, images, labels, batch_size,
            graph=self._graph, reuse=reuse)
        return self._net.loss()

    def train(self):
        with self._graph.as_default(), tf.device('/cpu:0'):
            self._setup_gradients()
            self._setup_train_operation()
            self._init_session()
            if tf.gfile.Exists(self._checkpoint_path):
                self._load_checkpoint()
            tf.train.start_queue_runners(sess=self._session)
            self._net.save_graph()
            # train iterations
            config = self.config.train
            for step in range(config.max_steps):
                self._step_time = time.time()
                _, loss_val = self._session.run([self._train_op, self._loss])
                if np.isnan(loss_val):
                    raise ValueError('Model diverged with loss = NaN')
                if step % 10 == 0:
                    self._update_progress(step, loss_val)
                if step % 100 == 0:
                    self._save_summary(step)
                if step % 5000 == 0 or (step + 1) == config.max_steps:
                    self._save_checkpoint(step)

    def _setup_gradients(self):
        config = self.config.train
        # ensure batch size is divisible by number of gpus
        if config.batch_size % config.num_gpus != 0:
            raise ValueError('Batch size must be divisible by number of GPUs')
        # initialize images and labels
        images_splits, labels_splits = self.preprocess()
        # for each gpu
        iterator = enumerate(zip(images_splits, labels_splits))
        tower_grads = []
        reuse = None
        for i, images_split, label_split in iterator:
            # loss with the proper nested contexts
            device_context = tf.device('/gpu:{}'.format(i))
            name_scope_context = tf.name_scope('tower_{}'.format(i))
            with device_context, name_scope_context:
                cpu_ctx = slim.arg_scope(
                    [slim.model_variable, slim.variable], device='/cpu:0')
                with cpu_ctx:
                    # loss from the final tower
                    self._loss = self.tower_loss(
                        images_split, label_split, reuse)
            reuse = True
            # batch norm updates from the final tower
            self._batch_norm_updates = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, name_scope_context)
            # gradients from all towers
            grads = self.optimizer.compute_gradients(self._loss)
            tower_grads.append(grads)
        self._gradients = _average_gradients(tower_grads)

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

    @property
    def _checkpoint_path(self):
        return self.config.name + '_' + self.config.dataset.name + '.ckpt'

    def _load_checkpoint(self):
        restorer = tf.train.Saver(tf.trainable_variables())
        restorer.restore(self._session, self._checkpoint_path)
        print('Pre-trained model restored from {}'.format(
            self._checkpoint_path))

    def _save_checkpoint(self, step):
        saver = tf.train.Saver(tf.global_variables())
        saver.save(self._session, self._checkpoint_path, global_step=step)

    def _save_summary(self, step):
        raise NotImplementedError

    def _update_progress(self, step, loss_val):
        duration = time.time() - self._step_time
        imgs_per_sec = self.config.train.batch_size / float(duration)
        now = datetime.now()
        info = '{}: step {}, loss = {:.2f} '.format(now, step, loss_val)
        info += '({:.1f} imgs/sec; {:.3f} sec/batch)'.format(
            imgs_per_sec, duration)
        print(info)
