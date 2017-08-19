import os
import copy
import glob

import yaml


def _dot_path(dictionary, dot_path_key):
    *dot_path, final_key = dot_path_key.split('.')
    for key in dot_path:
        dictionary = dictionary[key]
    return dictionary, final_key


class _DotDict(dict):
    def __init__(self, data):
        super().__init__(data)

    def _recursive_apply(self, obj, func_map):
        if isinstance(obj, dict):
            for k, v in obj.items():
                obj[k] = self._recursive_apply(v, func_map)
        elif isinstance(obj, (tuple, list, set, frozenset)):
            obj = obj.__class__(
                [self._recursive_apply(v, func_map) for v in obj])
        for cls, func in func_map.items():
            if isinstance(obj, cls):
                return func(obj)
        return obj

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

    def __getitem__(self, key):
        obj, key = _dot_path(self, key)
        return super(_DotDict, obj).__getitem__(key)

    def __setitem__(self, key, value):
        obj, key = _dot_path(self, key)
        return super(_DotDict, obj).__setitem__(key, value)

    def __delitem__(self, key):
        obj, key = _dot_path(self, key)
        return super(_DotDict, obj).__delitem__(key)

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Config(_DotDict):
    def __init__(self, yaml_files, overrides=None):
        unified = {}
        dictionary = {}
        mode = 'validation'
        for path in yaml_files:
            with open(path, 'r') as file:
                d = yaml.load(file.read())
            unified.update(copy.deepcopy(d))
            if 'dataset' in d:
                unified['dataset']['path'].setdefault(
                    'root', os.path.dirname(path))
                self._init_dataset(path, d)
            if 'train' in d:
                mode = 'train'
            dictionary.update(d)
        dictionary['mode'] = mode
        super().__init__(dictionary)
        self._setup_excepthook()
        self._init_overrides(dictionary, overrides)
        self._wrap(self)
        self._link(self)
        self._init_overrides(unified, overrides)
        self.unified = unified

    @staticmethod
    def _init_overrides(dictionary, overrides):
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
            sub_dictionary, k = _dot_path(dictionary, k)
            sub_dictionary[k] = v

    @staticmethod
    def _init_dataset(path, d):
        d = d['dataset']
        try:
            root = d['path'].pop('root')
        except KeyError:
            root = os.path.dirname(path)
        # change relative path to our working directory
        for mode, path in d['path'].items():
            d['path'][mode] = os.path.join(root, path)
        # add an unlabelled class to num_classes
        d['num_classes'] += 1

    def to_yaml(self, file=None):
        if file is not None:
            file = open(file, 'w')
        unified = self._recursive_apply(
            self.unified, {dict: lambda o: dict(o)})
        return yaml.dump(unified, file)

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
