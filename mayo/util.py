import os
import functools
import collections
from importlib.util import spec_from_file_location, module_from_spec

import numpy as np
import tensorflow as tf


def memoize_method(func):
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


_persistent_dict = {}


def delta(name, value):
    name += '.delta'
    prev_value = _persistent_dict.get(name, value)
    _persistent_dict[name] = value
    return value - prev_value


def every(name, value, interval):
    if interval <= 0:
        return False
    name += '.every'
    next_value = _persistent_dict.setdefault(name, value) + interval
    if value < next_value:
        return False
    _persistent_dict[name] = value
    return True


def moving_metrics(name, value, std=True, over=100):
    name += '.moving'
    history = _persistent_dict.setdefault(name, [])
    while len(history) >= over:
        history.pop(0)
    history.append(value)
    mean = np.mean(history)
    if not std:
        return mean
    return mean, np.std(history)


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
            'Unable to find module {!r} in path {!r}.'.format(name, path))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_from_dot_path(path, m=None):
    components = path.split('.')
    if m is None:
        root = components.pop(0)
        try:
            m = __builtins__[root]
        except KeyError:
            m = __import__(root)
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


def multi_objects_from_params(params, import_from=None, import_from_prefix=''):
    if not isinstance(params, collections.Sequence):
        params = [params]
    return [
        object_from_params(p, import_from, import_from_prefix)
        for p in params]


def format_shape(shape):
    return ' x '.join(str(s) if s else '?' for s in shape)


def format_percent(value):
    return '{:.2f}%'.format(value * 100)


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


def format_info(info, count=None):
    count = count or {}
    header = [h for h in info[0]._fields]
    table = [header + [c for c in count.values()]] + ['-']
    totals = {k: 0 for k in count}
    for row in info[1:]:
        new_row = []
        row_totals = {k: 0 for k in count}
        for col, item in enumerate(row):
            if isinstance(item, tf.Variable):
                item = item.name
            if isinstance(item, tf.TensorShape):
                total = item.num_elements()
                item = format_shape(item)
            if isinstance(item, (int, float)):
                total = item
            count_name = header[col]
            if count_name in count:
                row_totals[count_name] = total
            new_row.append(item)
        for name in count:
            total = row_totals[name]
            new_row.append(total)
            totals[name] += total
        table.append(new_row)
    if totals:
        table += ['-']
        total_row = [None] * (len(header) - 1) + ['    Total:']
        total_row += list(totals.values())
        table += [total_row]
    return tabular(table)


def unique(items):
    found = set()
    keep = []
    for item in items:
        if item in found:
            continue
        found.add(item)
        keep.append(item)
    return keep


def flatten(items, skip_none=False):
    for i in items:
        if isinstance(i, (list, tuple)):
            yield from flatten(i)
        elif i is not None:
            yield i
