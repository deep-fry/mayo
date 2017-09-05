import os
import re

import yaml
import tensorflow as tf

from mayo.log import log
from mayo.util import format_shape


class CheckpointHandler(object):
    _checkpoint_basename = 'checkpoint'

    def __init__(
            self, session, net_name, dataset_name,
            load, save, search_paths):
        super().__init__()
        self._session = session
        self._net = net_name
        self._dataset = dataset_name
        self._load, self._save = load, save
        self._load_latest = not isinstance(self._load, int)
        self._search_paths = search_paths

    def _variables(self, path=None):
        with self._session.graph.as_default():
            global_vars = tf.global_variables()
        if not path:
            return global_vars
        reader = tf.train.NewCheckpointReader(path)
        var_shape_map = reader.get_variable_to_shape_map()
        restore_vars = []
        for v in global_vars:
            base_name, _ = v.name.split(':')
            shape = var_shape_map.get(base_name, None)
            if shape is None:
                log.warn(
                    'Variable named {!r} does not exist in checkpoint.'
                    .format(base_name))
                continue
            v_shape = v.shape.as_list()
            if shape != v_shape:
                msg = ('Variable named {!r} has shape ({}) mismatch with the '
                       'shape ({}) in checkpoint, not loading it.')
                msg = msg.format(
                    base_name, format_shape(v_shape), format_shape(shape))
                log.warn(msg)
                continue
            restore_vars.append(v)
        log.debug(
            'Checkpoint variables to restore: {}.'
            .format(', '.join(v.name for v in restore_vars)))
        return restore_vars

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

    def _load_path(self):
        cp_path = self._path(False)
        if self._load_latest:
            try:
                with open(cp_path, 'r') as f:
                    manifest = yaml.load(f)
            except FileNotFoundError:
                return 0, None
            cp_name = manifest['model_checkpoint_path']
            step = re.findall(self._checkpoint_basename + '-(\d+)', cp_name)
            step = int(step[0])
        else:
            cp_name = '{}-{}'.format(self._checkpoint_basename, self._load)
            step = self._load
        cp_dir = os.path.dirname(cp_path)
        path = os.path.join(cp_dir, cp_name)
        log.info('Loading {}checkpoint from {!r}...'.format(
            'latest ' if self._load_latest else '', path))
        if not os.path.exists(path + '.index'):
            raise FileNotFoundError(
                'Checkpoint named {!r} not found.'.format(path))
        return step, path

    def load(self):
        if not self._load:
            return 0
        step, path = self._load_path()
        if not path:
            return step
        restorer = tf.train.Saver(self._variables(path))
        restorer.restore(self._session, path)
        log.info('Pre-trained model restored from {}'.format(path))
        return step

    def save(self, step):
        if not self._save:
            return
        log.info('Saving checkpoint at step {}...'.format(step))
        cp_path = self._path(True)
        saver = tf.train.Saver(self._variables())
        saver.save(self._session, cp_path, global_step=step)
