import tensorflow as tf
from tensorflow.contrib import slim

from model_slim.preprocessing import inception_preprocessing

from mayo.util import memoize


class Train(object):
    def __init__(self, config, net):
        super().__init__()
        self.config = config
        self.net = net

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
        rate = self.config.train.initial_learning_rate
        step = self.global_step
        batches_per_epoch = self.config.dataset.num_examples_per_epoch.train
        batches_per_epoch /= self.config.dataset.batch_size
        decay_steps = int(
            batches_per_epoch * self.config.train.num_epochs_per_decay)
        decay_factor = self.config.train.learning_rate_decay_factor
        return tf.train.exponential_decay(
            rate, step, decay_steps, decay_factor, staircase=True)

    @property
    @memoize
    def optimizer(self):
        params = self.config.train.optimizer
        Optimizer = getattr(tf.train, params.pop('type'))
        return Optimizer(self.learning_rate, **params)

    def train(self):
        with self.net.graph.as_default(), tf.device('/cpu:0'):
            self._train()

    def _train(self):

    @memoize
    def train_op(self):
        train = tf.contrib.layers.optimize_loss(
            loss=self.loss_op(),
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=self.config['learning_rate'],
            optimizer=self.config.get('optimizer', 'SGD'))
        self.end_points['train'] = train
        return train
