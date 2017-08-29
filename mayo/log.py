import os
import sys
from contextlib import contextmanager


class Logger(object):
    _levels = {
        'debug': 0,
        'info': 1,
        'warn': 2,
        'error': 3,
        'off': 4,
    }
    _colors = {
        'debug': '\033[100m',
        'info': '\033[44m',
        'warn': '\033[43m',
        'error': '\033[41m',
    }
    _signs = {
        'debug': ' ',
        'info': '.',
        'warn': '!',
        'error': '‼',
    }
    _colors_end = '\033[0m'

    def __init__(self):
        super().__init__()
        self._level = self._levels['info']
        self.pause_level = self._levels['error']
        self.color = 'color' in os.environ['TERM']
        self.width = 80
        self._last_is_update = False

    @property
    def level(self):
        for k, v in self._levels.items():
            if v == self._level:
                return v
        raise ValueError('Unrecognized log level.')

    @level.setter
    def level(self, value):
        self._level = self._levels[value]
        self.debug('Log level: {}'.format(value))

    def debug(self, text, update=False):
        return self.log(text, 'debug', update)

    def info(self, text, update=False):
        return self.log(text, 'info', update)

    def warn(self, text, update=False):
        return self.log(text, 'warn', update)

    def error(self, text, update=False):
        return self.log(text, 'error', update)

    @contextmanager
    def use_level(self, level):
        self._prev_level = self._level
        self._level = self._levels[level]
        yield
        self._level = self._prev_level

    def _header(self, text, level):
        if not self.color:
            color = colors_end = ''
        else:
            color = self._colors[level] + '\033[97m'
            colors_end = self._colors_end
        sign = self._signs[level]
        return '{}{}{} {}'.format(color, sign, colors_end, text)

    def log(self, text, level='info', update=False):
        num_level = self._levels[level]
        if self._level > num_level:
            return
        if update:
            begin = '\r'
            end = ''
            header_len = 2
            text = text[:min(self.width - header_len, len(text))]
        else:
            begin = ''
            end = '\n'
        text = self._header(text, level)
        if not update and self._last_is_update:
            begin = '\n' + begin
        print(begin + text, end=end)
        self._last_is_update = update
        while num_level >= self.pause_level:
            r = input(
                'Continue [Return], Stack trace [t], '
                'Debugger [d], Abort [q]: ')
            if not r:
                break
            elif r == 'd':
                import ipdb
                ipdb.set_trace()
            elif r == 't':
                import traceback
                traceback.print_stack()
            elif r == 'q':
                sys.exit(-1)


log = Logger()
