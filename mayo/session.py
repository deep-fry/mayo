import os
import re
import subprocess
from contextlib import contextmanager

import tensorflow as tf

from mayo.log import log
from mayo.net import Net
from mayo.util import memoize_method, memoize_property, Change, flatten, Table
from mayo.override import ChainOverrider
from mayo.checkpoint import CheckpointHandler
from mayo.preprocess import Preprocess


class Session(object):
    mode = None
    concurrent = False

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.change = Change()
        self._init_gpus()
        self.graph = tf.Graph()
        self.tf_session = tf.Session(
            graph=self.graph,
            config=tf.ConfigProto(allow_soft_placement=True))
        self.preprocessor = Preprocess(
            self.tf_session, self.mode, self.concurrent, config)
        self.checkpoint = CheckpointHandler(
            self.tf_session, config.system.search_path.checkpoint)
        self.nets = self._instantiate_nets()
        self.init_vars()

    def __del__(self):
        log.debug('Finishing...')
        del self.preprocessor
        self.tf_session.close()

    @property
    def num_gpus(self):
        return self.config.system.num_gpus if self.concurrent else 1

    def _auto_select_gpus(self):
        mem_bound = 500
        try:
            info = subprocess.check_output(
                'nvidia-smi', shell=True, stderr=subprocess.STDOUT)
            info = re.findall('(\d+)MiB\s/', info.decode('utf-8'))
            log.debug('GPU memory usages (MB): {}'.format(', '.join(info)))
            info = [int(m) for m in info]
            gpus = [i for i in range(len(info)) if info[i] <= mem_bound]
        except subprocess.CalledProcessError:
            gpus = []
        if len(gpus) < self.num_gpus:
            log.warn(
                'Number of GPUs available {} is less than the number of '
                'GPUs requested {}.'.format(len(gpus), self.num_gpus))
        return ','.join(str(g) for g in gpus[:self.num_gpus])

    def _init_gpus(self):
        gpus = self.config.system.get('visible_gpus', 'auto')
        if gpus != 'auto':
            if isinstance(gpus, list):
                gpus = ','.join(str(g) for g in gpus)
            else:
                gpus = str(gpus)
        else:
            gpus = self._auto_select_gpus()
        if gpus:
            log.info('Using GPUs: {}'.format(gpus))
        else:
            log.info('Not using GPUs.')
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    @contextmanager
    def as_default(self):
        with self.tf_session.as_default():
            with self.tf_session.graph.as_default():
                yield

    def _tf_int(self, name, dtype=tf.int64):
        with self.as_default():
            return tf.get_variable(
                name, [], initializer=tf.constant_initializer(0),
                trainable=False, dtype=dtype)

    @memoize_property
    def imgs_seen(self):
        return self._tf_int('imgs_seen', tf.int64)

    @memoize_property
    def num_steps(self):
        return self.imgs_seen / self.config.system.batch_size

    @memoize_property
    def num_epochs(self):
        imgs_per_epoch = self.config.dataset.num_examples_per_epoch.train
        return self.imgs_seen / imgs_per_epoch

    def _mean_metric(self, func):
        with self.as_default():
            return tf.reduce_mean(list(self.net_map(func)))

    @memoize_property
    def accuracy(self):
        return self._mean_metric(lambda net: net.accuracy())

    @memoize_property
    def loss(self):
        return self._mean_metric(lambda net: net.loss())

    def global_variables(self):
        with self.as_default():
            return tf.global_variables()

    def trainable_variables(self):
        with self.as_default():
            return tf.trainable_variables()

    def moving_average_variables(self):
        with self.as_default():
            return tf.moving_average_variables()

    def get_collection(self, key):
        func = lambda net: tf.get_collection(key)
        return flatten(self.net_map(func))

    @memoize_method
    def moving_average_op(self):
        decay = self.config.get('train.moving_average_decay', 0)
        if not decay:
            return None
        # instantiate moving average if moving_average_decay is supplied
        with self.as_default():
            var_avgs = tf.train.ExponentialMovingAverage(
                self.config.train.moving_average_decay, self.num_steps)
            avg_vars = tf.trainable_variables() + tf.moving_average_variables()
            return var_avgs.apply(avg_vars)

    def init_vars(self):
        log.debug('Initializing...')
        with self.as_default():
            return self.run(tf.variables_initializer(self.global_variables()))

    def info(self):
        return self.nets[0].info()

    def overrider_info(self):
        def flatten(overriders):
            for o in overriders:
                if isinstance(o, ChainOverrider):
                    yield from flatten(o)
                else:
                    yield o
        overrider_info = {}
        for o in flatten(self.nets[0].overriders):
            info = o.info(self)
            table = overrider_info.setdefault(o.__class__, Table(info._fields))
            table.add_row(info)
        return overrider_info

    def run(self, ops):
        return self.tf_session.run(ops)

    def _preprocess(self):
        with self.as_default():
            return self.preprocessor.preprocess(self.num_gpus)

    @contextmanager
    def _gpu_context(self, gid):
        with self.as_default():
            with tf.device('/gpu:{}'.format(gid)):
                with tf.name_scope('tower_{}'.format(gid)) as scope:
                    yield scope

    def _instantiate_nets(self):
        log.debug('Instantiating...')
        # ensure batch size is divisible by number of gpus
        if self.config.system.batch_size % self.num_gpus != 0:
            raise ValueError(
                'Batch size must be divisible by number of devices')
        is_training = self.mode == 'train'
        nets = []
        for i, (images, labels) in enumerate(self._preprocess()):
            log.debug('Instantiating graph for GPU #{}...'.format(i))
            with self._gpu_context(i):
                net = Net(self.config, images, labels, is_training, bool(nets))
            nets.append(net)
        return nets

    def net_map(self, func):
        for i, net in enumerate(self.nets):
            with self._gpu_context(i):
                yield func(net)

    def interact(self):
        from IPython import embed
        embed()
