import collections

import tensorflow as tf


def format_shape(shape):
    return ' x '.join(str(s) if s else '?' for s in shape)


def print_variables(description, variables, level):
    from mayo.log import log
    log_func = getattr(log, level)
    skipped = ['RMSProp', 'ExponentialMovingAverage']
    variables = [v for v in variables if not any(n in v for n in skipped)]
    if not variables:
        return
    if log.is_enabled('debug'):
        show_vars = '\n    '.join(variables)
        log_func('{}:\n    {}'.format(description, show_vars))
        return
    show_count = 10
    show_vars = '\n    '.join(variables[:show_count])
    num_more_vars = len(variables[show_count:])
    more_vars = ''
    if num_more_vars > 0:
        more_vars = '\n    ... '
        more_vars += '[{} more variables, use debug level to see all.]'
        more_vars = more_vars.format(num_more_vars)
    log_func('{}:\n    {}{}'.format(description, show_vars, more_vars))


class Percent(float):
    def __format__(self, _):
        return '{:.2f}%'.format(self * 100)


class Unknown(object):
    def __add__(self, other):
        return other
    __radd__ = __add__

    def __str__(self):
        return ''


unknown = Unknown()


class Table(collections.Sequence):
    def __init__(self, headers, formatters=None):
        super().__init__()
        self._headers = list(headers)
        self._empty_formatters = {h: None for h in headers}
        self._formatters = formatters or self._empty_formatters
        self._rows = []
        self._rules = []
        self._footers = {}
        self.add_rule()

    def __getitem__(self, index):
        if isinstance(index, tuple):
            row, col = index
            return self._rows[row][self._headers.index(col)]
        return self._rows[index]

    def __len__(self):
        return len(self._rows)

    @property
    def num_columns(self):
        return len(self._headers)

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
            row = [row[h] for h in self._headers]
        if len(row) != len(self._headers):
            raise ValueError(
                'Number of columns of row {!r} does not match headers {!r}.'
                .format(row, self._headers))
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

    def footer_sum(self, column):
        self._footers[column] = {'method': 'sum'}

    def footer_mean(self, column, weights=None):
        self._footers[column] = {'method': 'mean', 'weights': weights}

    def get_column(self, name):
        try:
            col_idx = self._headers.index(name)
        except ValueError:
            raise KeyError('Unable to find header named {!r}.'.format(name))
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
        elif isinstance(value, (list, tuple)):
            value = ', '.join(self._format_value(v) for v in value)
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

    def _get_footers(self):
        footer = [None] * self.num_columns
        for column, prop in self._footers.items():
            index = self._headers.index(column)
            column = self.get_column(column)
            if prop['method'] == 'sum':
                value = sum(column)
            elif prop['method'] == 'mean':
                try:
                    weights = self.get_column(prop.get('weights'))
                    if isinstance(weights, str):
                        weights = self.get_column(weights)
                except KeyError:
                    weights = [1] * len(column)
                value = sum(
                    v * w for v, w in zip(column, weights)
                    if unknown not in (v, w))
                value /= sum(w for w in weights if w is not unknown)
                if any(isinstance(c, Percent) for c in column):
                    value = Percent(value)
            else:
                raise TypeError('Unrecognized method.')
            footer[index] = value
        return footer

    def _plumb_value(self, value):
        if value is None or value is unknown:
            return
        if isinstance(value, Percent):
            return float(value)
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, (list, tuple)):
            return [self._plumb_value(v) for v in value]
        if isinstance(value, dict):
            return {
                self._plumb_value(k): self._plumb_value(v)
                for k, v in value.items()}
        if isinstance(value, tf.Variable):
            return value.name
        if isinstance(value, tf.TensorShape):
            return [int(s) for s in value]
        return str(value)

    def plumb(self):
        infos = {'items': []}
        for row in self._rows:
            info = self._plumb_value({
                k: v for k, v in zip(self._headers, row)})
            infos['items'].append(info)
        if self._footers:
            footer = self._get_footers()
            infos['footer'] = self._plumb_value({
                key: value for key, value in zip(self._headers, footer)
                if value is not None})
        return infos

    def _footer_row(self, widths=None):
        if not self._footers:
            return
        footer = self._get_footers()
        return self._format_row(footer, self._empty_formatters, widths=widths)

    def _column_widths(self):
        row_widths = []
        others = [self._headers, self._footer_row()]
        for row in self._rows + others:
            if row is None:
                continue
            if len(row) != self.num_columns:
                raise ValueError(
                    'Number of columns in row {} does not match headers {}.'
                    .format(row, self._headers))
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
        # footer
        footer = self._footer_row(widths)
        if footer:
            self.add_rule()
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

    def csv(self):
        rows = [', '.join(self._headers)]
        for r in self._rows:
            row = []
            for v in r:
                if isinstance(v, Percent):
                    v = float(v)
                row.append(str(v))
            rows.append(', '.join(row))
        return '\n'.join(rows)
