import os
import re
import glob

import yaml
import tensorflow as tf

from mayo.log import log
from mayo.util import format_shape


class CheckpointHandler(object):
    _checkpoint_basename = 'checkpoint'

    def __init__(self, session, load, save, search_paths):
        super().__init__()
        self._session = session
        self._load, self._save = load, save
        self._search_paths = search_paths
        self._checkpoint_directories = {}

    def _variables(self, vars_to_restore=None, path=None):
        with self._session.graph.as_default():
            vars_to_restore = vars_to_restore or tf.global_variables()
        if not path:
            return vars_to_restore
        reader = tf.train.NewCheckpointReader(path)
        var_shape_map = reader.get_variable_to_shape_map()
        restore_vars = []
        for v in vars_to_restore:
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

    def _directory(self, is_saving):
        try:
            return self._checkpoint_directories[is_saving]
        except KeyError:
            pass
        first_directory = None
        paths = self._search_paths
        paths = paths.save if is_saving else paths.load
        for path in paths:
            first_directory = first_directory or path
            has = os.path.isdir(path)
            if has:
                cp_path = os.path.join(path, self._checkpoint_basename + '-*')
                if glob.glob(cp_path):
                    self._checkpoint_directory = path
                    return path
        self._checkpoint_directories[is_saving] = first_directory
        return first_directory

    def _path(self, is_saving):
        directory = self._directory(is_saving)
        log.debug('Using {!r} for checkpoints.'.format(directory))
        if is_saving:
            # ensure directory exists
            os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, self._checkpoint_basename)

    def _load_path(self):
        cp_path = self._path(False)
        if self._load == 'latest':
            try:
                with open(cp_path, 'r') as f:
                    manifest = yaml.load(f)
            except FileNotFoundError:
                return 0, None
            cp_name = manifest['model_checkpoint_path']
            step = re.findall(self._checkpoint_basename + '-(\d+)', cp_name)
            step = int(step[0])
        elif self._load == 'pretrained':
            cp_name = 'pretrained'
            step = 0
        else:
            cp_name = '{}-{}'.format(self._checkpoint_basename, self._load)
            step = self._load
        cp_dir = os.path.dirname(cp_path)
        path = os.path.join(cp_dir, cp_name)
        load_name = ''
        if not isinstance(self._load, int):
            load_name = self._load + ' '
        log.info('Loading {}checkpoint from {!r}...'.format(load_name, path))
        if not os.path.exists(path + '.index'):
            raise FileNotFoundError(
                'Checkpoint named {!r} not found.'.format(path))
        return step, path

    def load(self, vars_to_restore=None):
        if not self._load and not isinstance(self._load, int):
            return 0
        step, path = self._load_path()
        if not path:
            return step
        vars_to_restore = self._variables(vars_to_restore, path)
        restorer = tf.train.Saver(vars_to_restore)
        restorer.restore(self._session, path)
        log.info('Pre-trained model restored.')
        return step

    def save(self, step):
        if not self._save:
            return
        log.info('Saving checkpoint at step {}...'.format(step))
        cp_path = self._path(True)
        saver = tf.train.Saver(self._variables())
        saver.save(self._session, cp_path, global_step=step)


class CheckpointSurgeon(object):
    def __init__(self, ckpt):
        super().__init__()
        self.ckpt = ckpt
        self._session = tf.Session()

    def var_to_shape_map(self, ckpt=None):
        return dict(tf.contrib.framework.list_variables(ckpt or self.ckpt))

    def var_to_var_map(self, rules):
        vvmap = {}
        for v in self.var_to_shape_map(self.ckpt):
            nv = v
            for pattern, replacement in rules.items():
                if replacement is None:
                    if re.findall(pattern, nv):
                        break
                else:
                    nv = re.sub(pattern, replacement, nv)
            else:
                vvmap[v] = nv
        return vvmap

    def _check_unassigned(self, renamed_vars, match_ckpt):
        if not match_ckpt:
            return
        log.info('Checking for unassigned variables...')
        to_vars = self.var_to_shape_map(match_ckpt)
        uninit_vars = [v for v in to_vars if v not in renamed_vars]
        if not uninit_vars:
            log.info('All variables are assigned.')
            return
        log.warn('Variables below will not be assigned:')
        for v in uninit_vars:
            log.warn('    - {}'.format(uninit_vars))

    def rename(self, to_ckpt, match_ckpt, rules, dry_run=False):
        log.info('Renaming variables...')
        vvmap = self.var_to_var_map(rules)
        new_vars = []
        for var_name, shape in self.var_to_shape_map(self.ckpt).items():
            try:
                new_var_name = vvmap[var_name]
            except KeyError:
                log.debug(
                    'Skipping {!r} as it is not required'.format(var_name))
                continue
            if new_var_name != var_name:
                log.debug(
                    'Renamed {!r} as {!r}.'.format(var_name, new_var_name))
            else:
                log.debug('{!r} is not renamed.'.format(var_name))
            with self._session.as_default():
                var = tf.contrib.framework.load_variable(self.ckpt, var_name)
                new_vars.append(tf.Variable(var, name=new_var_name))
        self._check_unassigned(vvmap.values(), match_ckpt)
        if dry_run:
            log.info('Dry run, not actually saving.')
            return
        log.info('Saving checkpoint with renamed variables...')
        saver = tf.train.Saver()
        self._session.run(tf.variables_initializer(new_vars))
        saver.save(self._session, to_ckpt)
