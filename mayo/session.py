import os
import re
import subprocess
from contextlib import contextmanager

import tensorflow as tf

from mayo.log import log
from mayo.util import memoize_method
from mayo.checkpoint import CheckpointHandler
from mayo.preprocess import Preprocess


class Session(object):
    multi_gpus = True

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._init_gpus()
        self.graph = tf.Graph()
        self.tf_session = tf.Session(
            graph=self.graph,
            config=tf.ConfigProto(allow_soft_placement=True))
        self.preprocessor = Preprocess(self.tf_session, config)
        self.checkpoint = CheckpointHandler(
            self.tf_session, config.system.search_path.checkpoint)

    def __del__(self):
        log.debug('Finishing...')
        del self.preprocessor
        self.tf_session.close()

    def _auto_select_gpus(self):
        mem_bound = 500
        num_gpus = self.config.system.num_gpus if self.multi_gpus else 1
        try:
            info = subprocess.check_output(
                'nvidia-smi', shell=True, stderr=subprocess.STDOUT)
            info = re.findall('(\d+)MiB\s/', info.decode('utf-8'))
            log.debug('GPU memory usage (MB): {}'.format(', '.join(info)))
            info = [int(m) for m in info]
            gpus = [i for i in range(len(info)) if info[i] <= mem_bound]
        except subprocess.CalledProcessError:
            gpus = []
        if len(gpus) < num_gpus:
            log.warn(
                'Number of GPUs available {} is less than the number of '
                'GPUs requested {}.'.format(len(gpus), num_gpus))
        return ','.join(str(g) for g in gpus[:num_gpus])

    def _init_gpus(self):
        gpus = self.config.system.get('visible_gpus', 'auto')
        if gpus != 'auto' and isinstance(gpus, list):
            gpus = ','.join(str(g) for g in gpus)
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

    @property
    @memoize_method
    def imgs_seen(self):
        return self._tf_int('imgs_seen', tf.int64)

    @property
    @memoize_method
    def num_steps(self):
        return self.imgs_seen / self.config.system.batch_size

    @property
    @memoize_method
    def num_epochs(self):
        imgs_per_epoch = self.config.dataset.num_examples_per_epoch.train
        return self.imgs_seen / imgs_per_epoch

    def global_variables(self):
        with self.as_default():
            return tf.global_variables()

    def trainable_variables(self):
        with self.as_default():
            return tf.trainable_variables()

    def moving_average_variables(self):
        with self.as_default():
            return tf.moving_average_variables()

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

    def init(self):
        log.debug('Initializing...')
        return self.run(tf.variables_initializer(self.global_variables()))

    def run(self, ops):
        log.debug('Running {!r}...'.format(ops))
        return self.tf_session.run(ops)

    def preprocess(self, mode):
        with self.as_default():
            if mode == 'train':
                return self.preprocessor.preprocess_train()
            elif mode == 'validate':
                return self.preprocessor.preprocess_validate()
        raise TypeError('Unrecognized mode {!r}'.format(mode))

    def interact(self):
        from IPython import embed
        embed()
