import tensorflow as tf

from mayo.log import log
from mayo.util import memoize_method
from mayo.checkpoint import CheckpointHandler
from mayo.preprocess import Preprocess


class Session(object):
    def __init__(self, config):
        super().__init__()
        self.session = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True))
        self.preprocessor = Preprocess(self.session, config)
        self.checkpoint = CheckpointHandler(
            self.session,
            config.system.checkpoint.load,
            config.system.checkpoint.save,
            config.system.search_paths.checkpoints)

    def __del__(self):
        log.info('Finishing...')
        del self.preprocessor
        tf.reset_default_graph()
        self.session.close()

    @staticmethod
    def _tf_int(name, dtype=tf.int64):
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
