import tensorflow as tf
from tensorflow.contrib import slim

from mayo.net import Net
from mayo.util import memoize, import_from_dot_path


class Train(object):
    def __init__(self, config):
        super().__init__()
        self.config = config

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
        raise NotImplementedError

    def tower_loss(self, images, labels, scope):
        net = Net(self.config, images, labels, batch_size)
        batch_size = images.get_shape().as_list()[0]

    def train(self):
        with self.net.graph.as_default(), tf.device('/cpu:0'):
            self._train()

    def _train(self):
        num_gpus = self.config.num_gpus
        images, labels = self.preprocess()
        split = lambda t: tf.split(
            axis=0, num_or_size_splits=num_gpus, value=t)
        images_splits = split(images)
        labels_splits = split(labels)
        iterator = enumerate(zip(images_splits, labels_splits))
        for i, images_split, label_split in iterator:
            device_context = tf.device('/gpu:{}'.format(i))
            name_scope_context = tf.name_scope('tower_{}'.format(i))
            with device_context, name_scope_context as scope:
                loss = self.tower_loss(images_split, label_split, scope)

    @memoize
    def train_op(self):
        train = tf.contrib.layers.optimize_loss(
            loss=self.loss_op(),
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=self.config['learning_rate'],
            optimizer=self.config.get('optimizer', 'SGD'))
        self.end_points['train'] = train
        return train
