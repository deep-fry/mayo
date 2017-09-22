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

    def __init__(self, session, load, save, search_path):
        super().__init__()
        self._session = session
        self._load, self._save = load, save
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

    def _epoch_path(self, epoch):
        name = '{}-{}'.format(self._checkpoint_basename, epoch)
        return os.path.join(self._directory(False), name)

    def list_epochs(self):
        files = self._directory_glob()
        checkpoints = []
        for f in files:
            c = os.path.splitext(f)[0]
            c = int(re.findall(self._checkpoint_basename + '-(\d+)', c)[0])
            if c not in checkpoints:
                checkpoints.append(c)
        return sorted(checkpoints)

    def _path(self, is_saving, load=None):
        directory = self._directory(is_saving)
        log.debug('Using {!r} for checkpoints.'.format(directory))
        if is_saving:
            # ensure directory exists
            os.makedirs(directory, exist_ok=True)
            return os.path.join(directory, self._checkpoint_basename)
        # loading
        if self._load == 'latest':
            manifest_file = os.path.join(directory, 'checkpoint')
            try:
                with open(manifest_file, 'r') as f:
                    manifest = yaml.load(f)
            except FileNotFoundError:
                raise CheckpointManifestNotFoundError(
                    'Manifest for the latest checkpoint cannot be found.')
            cp_name = manifest['model_checkpoint_path']
        elif self._load == 'pretrained':
            cp_name = self._load
        elif self._load == 'baseline':
            cp_name = self._load
        elif isinstance(self._load, int):
            cp_name = '{}-{}'.format(self._checkpoint_basename, self._load)
        else:
            raise ValueError(
                'Key "system.checkpoint.load" accepts either "baseline",'
                '"latest", "pretrained" or an epoch number.')
        path = os.path.join(directory, cp_name)
        load_name = ''
        if not isinstance(self._load, int):
            load_name = self._load + ' '
        log.info('Loading {}checkpoint from {!r}...'.format(load_name, path))
        if not os.path.exists(path + '.index'):
            raise CheckpointNotFoundError(
                'Checkpoint named {!r} not found.'.format(path))
        return path

    def _global_variables(self):
        with self._session.graph.as_default():
            return tf.global_variables()

    def load(self, epoch=None):
        if epoch is None and self._load is False:
            log.debug('Checkpoint loading disabled.')
            return
        if epoch is not None:
            path = self._epoch_path(epoch)
        else:
            try:
                path = self._path(False)
            except CheckpointManifestNotFoundError as e:
                log.warn('{} Abort load.'.format(e))
                return
        reader = tf.train.NewCheckpointReader(path)
        var_shape_map = reader.get_variable_to_shape_map()
        restore_vars = []
        for v in self._global_variables():
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
        restorer = tf.train.Saver(restore_vars)
        restorer.restore(self._session, path)
        log.debug('Checkpoint restored.')

    def save(self, epoch):
        if not self._save:
            return
        cp_path = self._path(True)
        if epoch == 'latest':
            log.info('Saving latest checkpoint to {!r}...'.format(cp_path))
        else:
            log.info(
                'Saving checkpoint at epoch {} to {!r}...'
                .format(epoch, cp_path))
        saver = tf.train.Saver(self._global_variables())
        step = 0 if epoch == 'latest' else epoch
        saver.save(self._session, cp_path, global_step=step)
