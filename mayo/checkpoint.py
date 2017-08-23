import os
import re

import yaml
import tensorflow as tf


class CheckpointHandler(object):
    _checkpoint_root = 'checkpoints/'
    _checkpoint_basename = 'checkpoint'

    def __init__(self, session, net_name, dataset_name):
        super().__init__()
        self._session = session
        self._net = net_name
        self._dataset = dataset_name

    @property
    def _checkpoint_path(self):
        directory = os.path.join(
            self._checkpoint_root, self._net, self._dataset)
        # ensure directory exists
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, self._checkpoint_basename)

    def _variables(self):
        with self._session.as_default():
            return tf.global_variables()

    def load(self):
        if not tf.gfile.Exists(self._checkpoint_path):
            return 0
        print('Loading latest checkpoint...')
        with open(self._checkpoint_path, 'r') as f:
            manifest = yaml.load(f)
        cp_name = manifest['model_checkpoint_path']
        cp_dir = os.path.dirname(self._checkpoint_path)
        path = os.path.join(cp_dir, cp_name)
        restorer = tf.train.Saver(self._variables())
        restorer.restore(self._session, path)
        print('Pre-trained model restored from {}'.format(path))
        step = re.findall(self._checkpoint_basename + '-(\d+)', cp_name)
        return int(step[0])

    def save(self, step):
        print('Saving checkpoint at step {}...'.format(step))
        saver = tf.train.Saver(self._variables())
        saver.save(self._session, self._checkpoint_path, global_step=step)
