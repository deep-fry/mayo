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
        return super(_DotDict, obj).__getitem__(key)

    def __setitem__(self, key, value):
        obj, key = self._dot_path(key)
        return super(_DotDict, obj).__setitem__(key, value)

    def __delitem__(self, key):
        obj, key = self._dot_path(key)
        return super(_DotDict, obj).__delitem__(key)

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Config(_DotDict):
    def __init__(self, yaml=None, path=None):
        if path:
            with open(path, 'r') as file:
                yaml = file.read()
        super().__init__(yamllib.load(yaml))
        self._setup_excepthook()
        self._init_dataset()

    def _init_dataset(self):
        dataset_path = self.dataset.path
        dataset_dir, _ = os.path.split(dataset_path)
        with open(dataset_path, 'r') as file:
            self.dataset = _DotDict(yamllib.load(file))
        # change relative path to our working directory
        for mode, path in self.dataset.path.items():
            self.dataset.path[mode] = os.path.join(dataset_dir, path)

    def image_shape(self):
        params = self.dataset.shape
        return (params.height, params.width, params.channels)

    def data_files(self, mode=None):
        mode = mode or self.mode
        try:
            path = self.dataset.path[mode]
        except KeyError:
            raise KeyError('Mode {} not recognized.'.format(mode))
        files = tf.gfile.Glob(path)
        if not files:
            msg = 'No files found for dataset {} with mode {} at {}'
            raise FileNotFoundError(msg.format(self.config.name, mode, path))
        return files

    def _setup_excepthook(self):
        import sys
        from IPython.core import ultratb
        use_pdb = self.get('use_pdb', False)
        sys.excepthook = ultratb.FormattedTB(call_pdb=use_pdb)
