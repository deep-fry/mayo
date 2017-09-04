import os
import functools
from importlib.util import spec_from_file_location, module_from_spec


def memoize(func):
    """
    A decorator to remember the result of the method call
    """
    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        name = '_' + func.__name__
        try:
            return getattr(self, name)
        except AttributeError:
            result = func(self, *args, **kwargs)
            setattr(self, name, result)
            return result
    return wrapped


@functools.lru_cache(maxsize=None)
def import_from_file(path):
    """
    Import module from path
    """
    name = os.path.split(path)[1]
    name = os.path.splitext(name)[0]
    spec = spec_from_file_location(name, path)
    if spec is None:
        raise ImportError(
            'Unable to find module "{}" in path "{}".'.format(name, path))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_from_dot_path(path, m=None):
    components = path.split('.')
    if m is None:
        m = __import__(components.pop(0))
    for c in components:
        m = getattr(m, c)
    return m


@functools.lru_cache(maxsize=None)
def import_from_string(string):
    if ':' in string:
        path, string = string.split(':')
        mod = import_from_file(path)
    else:
        mod = None
    return import_from_dot_path(string, mod)


def object_from_params(params, import_from=None, import_from_prefix=''):
    """
    import an object or get an object from <import_from> for <params>
    with YAML format:
        type: <importable object>
        # followed by arguments to create the object
        <argument>: <value>
        ...

    returns the imported object and params to create it with, which you can
    modify before using it as key-word arguments.
    """
    params = dict(params)
    try:
        otype = params.pop('type')
    except KeyError:
        raise KeyError(
            'Type of the object to create is missing '
            'from the parameter dictionary.')
    if import_from is not None:
        otype = import_from_prefix + otype
        try:
            cls = getattr(import_from, otype)
        except AttributeError:
            raise NotImplementedError(
                '{} does not implement object named {!r}'
                .format(import_from, otype))
    else:
        try:
            cls = import_from_string(otype)
        except ImportError:
            raise ImportError('Unable to import {!r}'.format(otype))
    for key in list(params):
        if key.startswith('_'):
            params.pop(key)
    return cls, params


def tabular(data):
    data = ['-'] + data + ['-']
    valid_rows = [row for row in data if row != '-']
    widths = [max(len(str(x)) for x in col if x) for col in zip(*valid_rows)]
    table = []
    for row in data:
        if row == '-':
            table.append('+-{}-+'.format('-+-'.join('-' * w for w in widths)))
            continue
        cols = []
        for width, x in zip(widths, row):
            if x is None:
                x = ''
            col = "{:{width}}".format(x, width=width)
            cols.append(col)
        table.append("| {} |".format(" | ".join(cols)))
    return '\n'.join(table)
