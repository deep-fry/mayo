from contextlib import contextmanager

import tensorflow as tf

from mayo.log import log
from mayo.util import memoize_method
from mayo.checkpoint import CheckpointHandler
from mayo.preprocess import Preprocess


class Session(object):
    def __init__(self, config):
        super().__init__()
        self.graph = tf.Graph()
        self.tf_session = tf.Session(
            graph=self.graph,
            config=tf.ConfigProto(allow_soft_placement=True))
        self.preprocessor = Preprocess(self.tf_session, config)
        self.checkpoint = CheckpointHandler(
            self.tf_session,
            config.system.checkpoint.load,
            config.system.checkpoint.save,
            config.system.search_paths.checkpoints)

    def __del__(self):
        log.debug('Finishing...')
        del self.preprocessor
        self.tf_session.close()

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
    def global_step(self):
        return self._tf_int('global_step', tf.int32)

    @property
    @memoize_method
    def imgs_seen(self):
        return self._tf_int('imgs_seen', tf.int64)

    def global_variables(self):
        with self.as_default():
            return tf.global_variables()

    def trainable_variables(self):
        with self.as_default():
            return tf.trainable_variables()

    def moving_average_variables(self):
        with self.as_default():
            return tf.moving_average_variables()

    def moving_average_op(self):
        decay = self.config.get('train.moving_average_decay', 0)
        if not decay:
            return None
        # instantiate moving average if moving_average_decay is supplied
        with self.as_default():
            var_avgs = tf.train.ExponentialMovingAverage(
                self.config.train.moving_average_decay, self.global_step)
            avg_vars = tf.trainable_variables() + tf.moving_average_variables()
            return var_avgs.apply(avg_vars)

    def init(self):
        log.debug('Initializing...')
        return self.run(tf.variables_initializer(self.global_variables()))

    def run(self, ops):
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
