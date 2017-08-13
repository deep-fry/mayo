import os

import tensorflow as tf
import yaml as yamllib


class _DotDict(dict):
    def __init__(self, data):
        super().__init__({})
        for key, value in data.items():
            self[key] = self._wrap(value)

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return value.__class__([self._wrap(v) for v in value])
        if isinstance(value, dict):
            return _DotDict(value)
        return value

    def _dot_path(self, dot_path_key):
        dictionary = self
        *dot_path, final_key = dot_path_key.split('.')
        for key in dot_path:
            dictionary = dictionary[key]
        return dictionary, final_key

    def __getitem__(self, key):
        obj, key = self._dot_path(key)
        return super(obj.__class__, obj).__getitem__(obj, key)

    def __setitem__(self, key, value):
        obj, key = self._dot_path(key)
        return super(obj.__class__, obj).__setitem__(obj, key, value)

    def __delitem__(self, key):
        obj, key = self._dot_path(key)
        return super(obj.__class__, obj).__delitem__(obj, key)

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Config(_DotDict):
    def __init__(self, yaml=None, path=None):
        if path:
            with open(path, 'r') as file:
                yaml = file.read()
        super().__init__(yamllib.load(yaml))

    @property
    def input_shape(self):
        params = self.dataset
        return (params.height, params.width, params.channels)

    def data_files(self, subset=None):
        subset = subset or self.config.dataset.subset
        path = self.config.dataset.data_dir
        pattern = os.path.join(path, '{}-*'.format(subset))
        data_files = tf.gfile.Glob(pattern)
        if not data_files:
            msg = 'No files found for dataset {} with subset {} at {}'
            raise FileNotFoundError(msg.format(self.config.name, subset, path))
        return data_files
