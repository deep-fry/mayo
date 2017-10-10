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


def memoize_property(func):
    return property(memoize_method(func))


class Change(object):
    def __init__(self):
        super().__init__()
        self._persistence = {}

    def delta(self, name, value):
        name += '.delta'
        prev_value = self._persistence.get(name, value)
        self._persistence[name] = value
        return value - prev_value

    def every(self, name, value, interval):
        if interval <= 0:
            return False
        name += '.every'
        next_value = self._persistence.setdefault(name, value) + interval
        if value < next_value:
            return False
        self._persistence[name] = value
        return True

    def moving_metrics(self, name, value, std=True, over=100):
        name += '.moving'
        history = self._persistence.setdefault(name, [])
        while len(history) >= over:
            history.pop(0)
        history.append(value)
        mean = np.mean(history)
        if not std:
            return mean
        return mean, np.std(history)

    def reset(self, name):
        for key in list(self._persistence):
            if not key.startswith(name + '.'):
                continue
            del self._persistence[key]


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
    return [
        object_from_params(p, import_from, import_from_prefix)
        for p in ensure_list(params)]


def format_shape(shape):
    return ' x '.join(str(s) if s else '?' for s in shape)


class Percent(float):
    def __format__(self, _):
        return '{:.2f}%'.format(self * 100)


class Table(collections.Sequence):
    def __init__(self, headers, formatters=None):
        super().__init__()
        self._headers = list(headers)
        self._empty_formatters = {h: None for h in headers}
        self._formatters = formatters or self._empty_formatters
        self._rows = []
        self._rules = []
        self._footers = None
        self.add_rule()

    def __getitem__(self, index):
        if isinstance(index, tuple):
            row, col = index
            return self._rows[row][self._headers.index(col)]
        return self._rows[index]

    def __len__(self):
        return len(self._rows)

    @classmethod
    def from_namedtuples(cls, tuples, formatters=None):
        table = cls(tuples[0]._fields, formatters)
        table.add_rows(tuples)
        return table

    @classmethod
    def from_dictionaries(cls, dictionaries, formatters=None):
        headers = list(dictionaries[0])
        table = cls(headers, formatters)
        for each in dictionaries:
            table.add_row([each[h] for h in headers])
        return table

    def add_row(self, row):
        if isinstance(row, collections.Mapping):
            row = (row[h] for h in self._headers)
        self._rows.append(list(row))

    def add_rows(self, rows):
        for row in rows:
            self.add_row(row)

    def add_rule(self):
        self._rules.append(len(self._rows))

    def add_column(self, name, func, formatter=None):
        for row_idx, row in enumerate(self._rows):
            new = func(row_idx)
            row.append(new)
        self._headers.append(name)
        self._formatters[name] = formatter

    def set_footer(self, footer):
        self._footers = footer

    def get_column(self, name):
        col_idx = self._headers.index(name)
        return [row[col_idx] for row in self._rows]

    def _format_value(self, value, formatter=None, width=None):
        if value is None:
            value = ''
        elif formatter:
            if isinstance(formatter, collections.Callable):
                value = formatter(value, width)
            else:
                value = formatter.format(value, width=width)
        elif isinstance(value, int):
            value = '{:{width},}'.format(value, width=width or 0)
        elif isinstance(value, float):
            value = '{:{width}.{prec}}'.format(
                value, width=width or 0, prec=3)
        elif isinstance(value, tf.Variable):
            value = value.name
        elif isinstance(value, tf.TensorShape):
            value = format_shape(value)
        else:
            value = str(value)
        if width:
            value = '{:{width}}'.format(value, width=width)
            if len(value) > width:
                value = value[:width - 1] + '…'
        return value

    def _format_row(self, row, formatters=None, widths=None):
        formatters = formatters or self._formatters
        widths = widths or [None] * len(self._headers)
        new_row = []
        for h, x, width in zip(self._headers, row, widths):
            x = self._format_value(x, formatters[h], width)
            new_row.append(x)
        return new_row

    def _column_widths(self):
        row_widths = []
        others = [self._headers, self._footers]
        for row in self._rows + others:
            if row is None:
                continue
            formatters = self._formatters
            if row in others:
                formatters = self._empty_formatters
            row = self._format_row(row, formatters)
            row_widths.append(len(e) for e in row)
        return [max(col) for col in zip(*row_widths)]

    def format(self):
        widths = self._column_widths()
        # header
        header = []
        for i, h in enumerate(self._headers):
            header.append(self._format_value(h, None, widths[i]))
        table = [header]
        # rows
        for row in self._rows:
            table.append(self._format_row(row, self._formatters, widths))
        if self._footers:
            self.add_rule()
            footer = self._format_row(
                self._footers, self._empty_formatters, widths)
            table.append(footer)
        # lines
        lined_table = []
        for row in table:
            row = " | ".join(
                e for e, h in zip(row, self._headers)
                if not h.endswith('_'))
            lined_table.append("| {} |".format(row))
        # rules
        rule = '+-{}-+'.format('-+-'.join(
            '-' * w for w, h in zip(widths, self._headers)
            if not h.endswith('_')))
        ruled_table = [rule]
        for index, row in enumerate(lined_table):
            ruled_table.append(row)
            if index in self._rules:
                ruled_table.append(rule)
        ruled_table.append(rule)
        return '\n'.join(ruled_table)


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


def ensure_list(str_or_list):
    if isinstance(str_or_list, str):
        return [str_or_list]
    if isinstance(str_or_list, list):
        return str_or_list
    raise TypeError('Unrecognized type.')
