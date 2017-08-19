import os
import glob

import yaml


class _DotDict(dict):
    def __init__(self, data):
        super().__init__(data)

    def _recursive_apply(self, obj, func_map):
        def apply(o):
            for t, func in func_map.items():
                if isinstance(o, t):
                    return func(o)
            return o
        if isinstance(obj, dict):
            for k, v in obj.items():
                obj[k] = self._recursive_apply(v, func_map)
        elif isinstance(obj, (tuple, list, set, frozenset)):
            obj = obj.__class__(
                [self._recursive_apply(v, func_map) for v in obj])
        return apply(obj)

    def _wrap(self, obj):
        def func(obj):
            if type(obj) is dict:
                return _DotDict(obj)
            return obj
        return self._recursive_apply(obj, {dict: func})

    def _link(self, obj):
        def func(obj):
            if obj[0] == '$':
                return self[obj[1:]]
            return obj
        return self._recursive_apply(obj, {str: func})

    def _dot_path(self, dot_path_key):
        dictionary = self
        *dot_path, final_key = dot_path_key.split('.')
        for key in dot_path:
            dictionary = dictionary[key]
        return dictionary, final_key

    def to_yaml(self):
        self_dict = self._recursive_apply(self, {dict: lambda o: dict(o)})
        return yaml.dump(self_dict)

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
    def __init__(self, yaml_files, overrides=None):
        dictionary = {}
        mode = 'validation'
        for path in yaml_files:
            with open(path, 'r') as file:
                d = yaml.load(file.read())
                if 'dataset' in d:
                    self._init_dataset(path, d)
                if 'train' in d:
                    mode = 'train'
                dictionary.update(d)
        dictionary['mode'] = mode
        super().__init__(dictionary)
        self._setup_excepthook()
        self._wrap(self)
        self._link(self)
        self._init_overrides(overrides)

    def _init_overrides(self, overrides):
        if not overrides:
            return
        for override in overrides.split(';'):
            k, v = (o.strip() for o in override.split('='))
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            self[k] = v

    @staticmethod
    def _init_dataset(path, d):
        root = os.path.dirname(path)
        # change relative path to our working directory
        paths = d['dataset']['path']
        for mode, path in paths.items():
            paths[mode] = os.path.join(root, path)
        # add an unlabelled class to num_classes
        d['dataset']['num_classes'] += 1

    def save(self, path):
        with open(path, 'w') as file:
            yaml.dump(self, file)

    def image_shape(self):
        params = self.dataset.shape
        return (params.height, params.width, params.channels)

    def data_files(self, mode=None):
        mode = mode or self.mode
        try:
            path = self.dataset.path[mode]
        except KeyError:
            raise KeyError('Mode {} not recognized.'.format(mode))
        files = glob.glob(path)
        if not files:
            msg = 'No files found for dataset {} with mode {} at {}'
            raise FileNotFoundError(msg.format(self.config.name, mode, path))
        return files

    def _excepthook(self, etype, evalue, etb):
        if isinstance(etype, KeyboardInterrupt):
            return
        from IPython.core import ultratb
        use_pdb = self.get('use_pdb', True)
        return ultratb.FormattedTB(call_pdb=use_pdb)(etype, evalue, etb)

    def _setup_excepthook(self):
        import sys
        sys.excepthook = self._excepthook
