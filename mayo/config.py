import os

import tensorflow as tf
import yaml


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
    def __init__(self, net, dataset=None, train=None):
        with open(net, 'r') as file:
            net_yaml = file.read()
        super().__init__(yaml.load(net_yaml))
        self._setup_excepthook()
        self._init_dataset(dataset)
        self._init_train(train)

    def _init_sub_config(self, name, path):
        with open(path, 'r') as file:
            self[name] = _DotDict(yaml.load(file))

    def _init_dataset(self, path):
        self._init_sub_config('dataset', path)
        root = os.path.dirname(path)
        # change relative path to our working directory
        for mode, path in self.dataset.path.items():
            self.dataset.path[mode] = os.path.join(root, path)

    def _init_train(self, path):
        if path is not None:
            self.mode = 'train'
            self._init_sub_config('train', path)
        else:
            self.mode = 'validation'

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
