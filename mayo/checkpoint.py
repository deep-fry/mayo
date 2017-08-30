import os
import re

import yaml
import tensorflow as tf

from mayo.log import log


class CheckpointHandler(object):
    _checkpoint_basename = 'checkpoint'

    def __init__(self, session, net_name, dataset_name, load, search_paths):
        super().__init__()
        self._session = session
        self._net = net_name
        self._dataset = dataset_name
        self._load = load
        self._search_paths = search_paths

    def _directory(self):
        try:
            return self._checkpoint_directory
        except AttributeError:
            pass
        first_directory = None
        for path in self._search_paths:
            directory = os.path.join(path, self._net, self._dataset)
            first_directory = first_directory or directory
            if os.path.isdir(directory):
                self._checkpoint_directory = directory
                return directory
        self._checkpoint_directory = first_directory
        return first_directory

    def _path(self, is_saving):
        directory = self._directory()
        if is_saving:
            # ensure directory exists
            os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, self._checkpoint_basename)

    def _variables(self):
        with self._session.as_default():
            return tf.global_variables()

    def load(self, must=False):
        if not self._load:
            return 0
        cp_path = self._path(False)
        try:
            with open(cp_path, 'r') as f:
                manifest = yaml.load(f)
        except FileNotFoundError:
            if must:
                raise
            return 0
        log.info('Loading latest checkpoint...')
        if isinstance(self._load, int):
            step = int(self._load)
            cp_name = '{}-{}'.format(self._checkpoint_basename, step)
        else:
            cp_name = manifest['model_checkpoint_path']
        cp_dir = os.path.dirname(cp_path)
        path = os.path.join(cp_dir, cp_name)
        restorer = tf.train.Saver(self._variables())
        restorer.restore(self._session, path)
        log.info('Pre-trained model restored from {}'.format(path))
        step = re.findall(self._checkpoint_basename + '-(\d+)', cp_name)
        return int(step[0])

    def save(self, step):
        if not self._save:
            return
        log.info('Saving checkpoint at step {}...'.format(step))
        cp_path = self._path(True)
        saver = tf.train.Saver(self._variables())
        saver.save(self._session, cp_path, global_step=step)
