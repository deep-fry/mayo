import re
import os
import ast
import sys
import glob
import operator
import collections

import yaml


class YamlTag(object):
    tag = None

    @classmethod
    def register(cls):
        yaml.add_constructor(cls.tag, cls.constructor)
        yaml.add_representer(cls, cls.representer)

    @classmethod
    def constructor(cls, loader, node):
        raise NotImplementedError

    @classmethod
    def representer(cls, dumper, data):
        raise NotImplementedError


class YamlScalarTag(YamlTag):
    def __init__(self, content):
        super().__init__()
        self.content = content

    @classmethod
    def constructor(cls, loader, node):
        content = loader.construct_scalar(node)
        return cls(content)

    @classmethod
    def representer(cls, dumper, tag):
        return dumper.represent_scalar(cls.tag, tag.content)

    def value(self):
        raise NotImplementedError

    def __repr__(self):
        return repr('{} {!r}'.format(self.tag, self.content))


class ArithTag(YamlScalarTag):
    tag = '!arith'
    _eval_expr_map = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.BitXor: operator.xor,
        ast.USub: operator.neg,
    }

    def value(self):
        tree = ast.parse(self.content, mode='eval').body
        return self._eval(tree)

    def _eval(self, n):
        if isinstance(n, ast.Num):
            return n.n
        if isinstance(n, ast.Call):
            op = self._eval(n.func)
            args = (self._eval(a) for a in n.args)
            return op(*args)
        if isinstance(n, ast.Attribute):
            return getattr(self._eval(n.value), n.attr)
        if isinstance(n, ast.Name):
            try:
                return __builtins__[n.id]
            except KeyError:
                return __import__(n.id)
        if not isinstance(n, (ast.UnaryOp, ast.BinOp)):
            raise TypeError('Unrecognized operator node {}'.format(n))
        op = self._eval_expr_map[type(n.op)]
        if isinstance(n, ast.UnaryOp):
            return op(self._eval(n.operand))
        return op(self._eval(n.left), self._eval(n.right))


class ExecTag(YamlScalarTag):
    tag = '!exec'

    def value(self):
        try:
            return self._value
        except AttributeError:
            pass
        variables = {}
        exec(self.content, variables)
        self._value = variables
        return variables


class EvalTag(YamlScalarTag):
    tag = '!eval'

    def value(self):
        try:
            return self._value
        except AttributeError:
            pass
        import tensorflow as tf
        import mayo
        self._value = eval(self.content, {'tf': tf, 'mayo': mayo})
        return self._value


ArithTag.register()
ExecTag.register()
EvalTag.register()


class _DotDict(collections.MutableMapping):
    def __init__(self, data, root=None, normalize=True):
        if not isinstance(data, collections.Mapping):
            raise TypeError(
                'Cannot construct {!r} from data of type {!r}'.format(
                    self.__class__, type(data)))
        super().__init__()
        super().__setattr__('_root', root or self)
        if not normalize:
            super().__setattr__('_mapping', data)
        else:
            super().__setattr__('_mapping', {})
            # normalize key paths
            for key, value in data.items():
                self[key] = value

    @classmethod
    def _merge(cls, d, md):
        for k, md_k in md.items():
            d_k = d.get(k)
            d_map = isinstance(d_k, collections.Mapping)
            md_map = isinstance(md_k, collections.Mapping)
            if d_map and md_map:
                cls._merge(d_k, md_k)
            else:
                d[k] = md_k

    def merge(self, md):
        self._merge(self, md)

    def _eval(self, value):
        if isinstance(value, YamlScalarTag):
            return value.__class__(self._eval(value.content)).value()
        if isinstance(value, str):
            regex = r'\$\(([_a-zA-Z][_a-zA-Z0-9\.]*)\)'
            while True:
                keys = re.findall(regex, value, re.MULTILINE)
                if not keys:
                    break
                for k in keys:
                    placeholder = '$({})'.format(k)
                    try:
                        v = str(self._root[k])
                    except KeyError:
                        raise KeyError(
                            'Attempting to resolve a non-existent key-path '
                            'with placeholder {!r}.'.format(placeholder))
                    value = value.replace(placeholder, v)
            return value
        if isinstance(value, collections.Mapping):
            if not isinstance(value, _DotDict):
                return _DotDict(value, self._root, False)
        if isinstance(value, (tuple, list, set, frozenset)):
            return value.__class__(self._eval(v) for v in value)
        return value

    def _dot_path(self, dot_path_key, dictionary=None, setdefault=None):
        def type_error(keyable, key):
            raise KeyError(
                'Key path {!r} resolution stopped at {!r} because the '
                'current object {!r} is not key-addressable.'
                .format(dot_path_key, key, keyable))

        *dot_path, final_key = dot_path_key.split('.')
        keyable = dictionary or self._mapping
        for index, key in enumerate(dot_path):
            try:
                if isinstance(keyable, (tuple, list)):
                    value = keyable[int(key)]
                elif isinstance(keyable, collections.Mapping):
                    if setdefault:
                        try:
                            next_key = dot_path[index + 1]
                        except IndexError:
                            next_key = final_key
                        default_cls = list if next_key.isdigit() else dict
                        value = keyable.setdefault(key, default_cls())
                    else:
                        value = keyable[key]
                else:
                    type_error(keyable, key)
            except (KeyError, IndexError):
                raise KeyError(
                    'Key path {!r} cannot be resolved.'.format(dot_path_key))
            keyable = value
        if isinstance(keyable, (tuple, list)):
            final_key = int(final_key)
        elif not isinstance(keyable, collections.Mapping):
            type_error(keyable, final_key)
        return keyable, final_key

    def __getitem__(self, key):
        obj, key = self._dot_path(key)
        return self._eval(obj[key])
    __getattr__ = __getitem__

    def __setitem__(self, key, value):
        obj, key = self._dot_path(key, setdefault=True)
        obj[key] = value
    __setattr__ = __setitem__

    def set(self, key, value):
        # old setattr behaviour
        super().__setattr__(key, value)

    def __delitem__(self, key):
        obj, key = self._dot_path(key)
        del obj[key]
    __delattr__ = __delitem__

    def __iter__(self):
        return iter(self._mapping)

    def __len__(self):
        return len(self._mapping)


class BaseConfig(_DotDict):
    def __init__(self, merge_hook=None):
        super().__init__({})
        self.set('_merge_hook', merge_hook or {})

    def merge(self, dictionary):
        super().merge(dictionary)
        for key, func in self._merge_hook.items():
            if key in _DotDict(dictionary):
                func()

    def yaml_update(self, file):
        with open(file, 'r') as f:
            dictionary = yaml.load(f)
        imports = dictionary.pop('_import', None)
        if imports:
            if isinstance(imports, str):
                imports = [imports]
            for i in imports:
                if not os.path.isabs(i):
                    i = os.path.join(os.path.dirname(file), i)
                self.yaml_update(i)
        self.merge(dictionary)

    def override_update(self, key, value):
        if isinstance(value, str):
            value = yaml.load(value)
        self.merge({key: value})

    def to_yaml(self, file=None):
        if file is not None:
            file = open(file, 'w')
        kwargs = {'explicit_start': True, 'width': 70, 'indent': 4}
        return yaml.dump(self._mapping, file, **kwargs)


class Config(BaseConfig):
    def __init__(self):
        merge_hook = {
            'system.log': self._setup_log_level,
        }
        super().__init__(merge_hook)
        self._setup_excepthook()
        self._init_system_config()
        self._check()

    def _init_system_config(self):
        root = os.path.dirname(__file__)
        self.yaml_update(os.path.join(root, 'system.yaml'))

    def _check(self):
        if os.environ.pop('CUDA_VISIBLE_DEVICES', None):
            from mayo import log
            log.warn(
                'Ignoring "CUDA_VISIBLE_DEVICES", as it is overridden '
                'by config "system.visible_gpus".')

    def image_shape(self):
        params = self.dataset.preprocess.shape
        return (params.height, params.width, params.channels)

    def label_offset(self):
        bg = self.dataset.background_class
        return int(bg.use) - int(bg.has)

    def num_classes(self):
        return self.dataset.num_classes + self.label_offset()

    def data_files(self, mode):
        path = self.dataset.path
        try:
            path = path[mode]
        except KeyError:
            raise KeyError('Mode {!r} not recognized.'.format(mode))
        files = []
        search_path = self.system.search_path.dataset
        paths = [path]
        if not os.path.isabs(path):
            paths = [os.path.join(d, path) for d in search_path]
        for p in paths:
            files += glob.glob(p)
        if not files:
            msg = 'No files found for dataset {!r} with mode {!r} at {!r}'
            raise FileNotFoundError(msg.format(
                self.dataset.name, mode, ', '.join(paths)))
        return files

    def _excepthook(self, etype, evalue, etb):
        from IPython.core import ultratb
        from mayo.util import import_from_string
        ultratb.FormattedTB()(etype, evalue, etb)
        for exc in self.get('system.pdb.skip', []):
            exc = import_from_string(exc)
            if issubclass(etype, exc):
                sys.exit(-1)
        if self.get('system.pdb.use', True):
            import ipdb
            ipdb.post_mortem(etb)

    def _setup_excepthook(self):
        sys.excepthook = self._excepthook

    def _setup_log_level(self):
        level = self.get('system.log.mayo', 'info')
        if level != 'info':
            from mayo.log import log
            log.level = level
        level = self.get('system.log.tensorflow', 0)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(level)
