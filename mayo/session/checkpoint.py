import os
import re
import glob

import yaml
import tensorflow as tf

from mayo.log import log
from mayo.util import format_shape


class CheckpointNotFoundError(FileNotFoundError):
    pass


class CheckpointManifestNotFoundError(FileNotFoundError):
    pass


class CheckpointHandler(object):
    _checkpoint_basename = 'checkpoint'
    _checkpoint_latest = 'latest'

    def __init__(self, session, search_path):
        super().__init__()
        self.tf_session = session
        self._search_path = search_path
        self._checkpoint_directories = {}

    def _directory(self, is_saving):
        try:
            return self._checkpoint_directories[is_saving]
        except KeyError:
            pass
        paths = self._search_path.get('save' if is_saving else 'load')
        path = paths[0]
        for each in paths:
            if not os.path.isdir(each):
                continue
            if self._directory_glob(each):
                path = each
                break
        self._checkpoint_directories[is_saving] = path
        return path

    def _directory_glob(self, directory=None):
        directory = directory or self._directory(False)
        return glob.glob(os.path.join(
            directory, self._checkpoint_basename + '-*'))

    def list_epochs(self):
        files = self._directory_glob()
        checkpoints = []
        for f in files:
            c = os.path.splitext(f)[0]
            c = int(re.findall(self._checkpoint_basename + '-(\d+)', c)[0])
            if c not in checkpoints:
                checkpoints.append(c)
        return sorted(checkpoints)

    def _path(self, key, is_saving):
        directory = self._directory(is_saving)
        log.debug('Using search path {!r} for checkpoints.'.format(directory))
        if isinstance(key, int):
            cp_name = '{}-{}'.format(self._checkpoint_basename, key)
        elif key == self._checkpoint_latest and not is_saving:
            manifest_file = os.path.join(directory, 'checkpoint')
            try:
                with open(manifest_file, 'r') as f:
                    manifest = yaml.load(f)
            except FileNotFoundError:
                raise CheckpointManifestNotFoundError(
                    'Manifest for the latest checkpoint cannot be found.')
            cp_name = manifest['model_checkpoint_path']
        else:
            cp_name = key
        if is_saving:
            # ensure directory exists
            os.makedirs(directory, exist_ok=True)
            return os.path.join(directory, cp_name)
        # loading
        path = os.path.join(directory, cp_name)
        log.info('Loading checkpoint from {!r}...'.format(path))
        if not os.path.exists(path + '.index'):
            raise CheckpointNotFoundError(
                'Checkpoint {!r} not found.'.format(path))
        return path

    def _global_variables(self):
        with self.tf_session.graph.as_default():
            return tf.global_variables()

    def load(self, key=_checkpoint_latest):
        if key is False or (key != 0 and not key):
            log.debug('Checkpoint loading disabled.')
            return []
        try:
            path = self._path(key, False)
        except CheckpointManifestNotFoundError as e:
            log.warn('{} Abort load.'.format(e))
            return []
        reader = tf.train.NewCheckpointReader(path)
        var_shape_map = reader.get_variable_to_shape_map()
        restore_vars = []
        missing_vars = []
        for v in self._global_variables():
            base_name, _ = v.name.split(':')
            shape = var_shape_map.get(base_name, None)
            if shape is None:
                missing_vars.append(base_name)
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
        if missing_vars:
            log.warn(
                'Variables missing in checkpoint:\n    {}'
                .format('\n    '.join(missing_vars)))
        log.debug(
            'Checkpoint variables to restore:\n    {}'
            .format('\n    '.join(v.name for v in restore_vars)))
        restorer = tf.train.Saver(restore_vars)
        restorer.restore(self.tf_session, path)
        log.debug('Checkpoint restored.')
        return restore_vars

    def save(self, key):
        cp_path = self._path(key, True)
        if isinstance(key, int):
            log.info(
                'Saving checkpoint at epoch {} to {!r}...'
                .format(key, cp_path))
        else:
            log.info('Saving checkpoint to {!r}...'.format(cp_path))
        saver = tf.train.Saver(self._global_variables())
        saver.save(self.tf_session, cp_path, write_meta_graph=False)
