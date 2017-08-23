import re
import os
import ast
import copy
import glob
import operator
import collections

import yaml


class ArithTag(object):
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

    def __init__(self, expr):
        super().__init__()
        self.expr = expr

    def value(self):
        tree = ast.parse(self.expr, mode='eval').body
        return self._eval(tree)

    def _eval(self, n):
        if isinstance(n, ast.Num):
            return n.n
        if isinstance(n, ast.Call):
            op = __builtins__[n.func.id]
            args = (self._eval(a) for a in n.args)
            return op(*args)
        if not isinstance(n, (ast.UnaryOp, ast.BinOp)):
            raise TypeError('Unrecognized operator node {}'.format(n))
        op = self._eval_expr_map[type(n.op)]
        if isinstance(n, ast.UnaryOp):
            return op(self._eval(n.operand))
        return op(self._eval(n.left), self._eval(n.right))

    @staticmethod
    def constructor(loader, node):
        expr = loader.construct_scalar(node)
        return ArithTag(expr)

    @classmethod
    def representer(cls, dumper, data):
        return dumper.represent_scalar(cls.tag, data.expr)

    def __repr__(self):
        return repr("{} '{}'".format(self.tag, self.expr))


yaml.add_constructor(ArithTag.tag, ArithTag.constructor)
yaml.add_representer(ArithTag, ArithTag.representer)


def _dot_path(keyable, dot_path_key):
    *dot_path, final_key = dot_path_key.split('.')
    for key in dot_path:
        if isinstance(keyable, (tuple, list)):
            key = int(key)
        keyable = keyable[key]
    return keyable, final_key


def _dict_merge(d, md):
    for k, v in md.items():
        can_merge = k in d and isinstance(d[k], dict)
        can_merge = can_merge and isinstance(md[k], collections.Mapping)
        if can_merge:
            _dict_merge(d[k], md[k])
        else:
            d[k] = md[k]


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
        def wrap(obj):
            if type(obj) is dict:
                return _DotDict(obj)
            return obj
        return self._recursive_apply(obj, {dict: wrap})

    def _link(self, obj):
        def link_str(string):
            keys = re.findall(r'\$\(([_a-zA-Z][_a-zA-Z0-9\.]+)\)', string)
            for k in keys:
                d, fk = _dot_path(obj, k)
                string = string.replace('$({})'.format(k), str(d[fk]))
            return string

        def link_arith(arith):
            try:
                return ArithTag(link_str(arith.expr)).value()
            except KeyError:
                return arith

        link_map = {str: link_str, ArithTag: link_arith}
        return self._recursive_apply(obj, link_map)

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
        self._setup_excepthook()
        unified = {}
        dictionary = {}
        self._init_system_config(unified, dictionary)
        for path in yaml_files:
            with open(path, 'r') as file:
                d = yaml.load(file)
            _dict_merge(unified, copy.deepcopy(d))
            if 'dataset' in d:
                self._init_dataset(path, unified, d)
            _dict_merge(dictionary, d)
        self._override(dictionary, overrides)
        self._link(dictionary)
        self._wrap(dictionary)
        super().__init__(dictionary)
        self._override(unified, overrides)
        self._link(unified)
        self.unified = unified

    @staticmethod
    def _init_dataset(path, u, d):
        root = os.path.dirname(path)
        u['dataset']['path'].setdefault('root', root)
        d = d['dataset']
        root = d['path'].pop('root', root)
        # change relative path to our working directory
        for mode, path in d['path'].items():
            d['path'][mode] = os.path.join(root, path)

    def _init_system_config(self, unified, dictionary):
        root = os.path.dirname(__file__)
        system_yaml = os.path.join(root, 'system.yaml')
        with open(system_yaml, 'r') as file:
            system = yaml.load(file)
        _dict_merge(unified, system)
        _dict_merge(dictionary, system)

    @staticmethod
    def _override(dictionary, overrides):
        if not overrides:
            return
        for override in overrides.split(';'):
            k, v = (o.strip() for o in override.split('='))
            sub_dictionary, k = _dot_path(dictionary, k)
            sub_dictionary[k] = yaml.load(v)

    def to_yaml(self, file=None):
        if file is not None:
            file = open(file, 'w')
        return yaml.dump(self.unified, file, width=80, indent=4)

    def image_shape(self):
        params = self.dataset.shape
        return (params.height, params.width, params.channels)

    def data_files(self, mode):
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
        use_pdb = self.system.use_pdb
        return ultratb.FormattedTB(call_pdb=use_pdb)(etype, evalue, etb)

    def _setup_excepthook(self):
        import sys
        sys.excepthook = self._excepthook
