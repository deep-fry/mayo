import os
import sys
import shutil
import itertools
from contextlib import contextmanager

from termcolor import colored


class Logger(object):
    _levels = {
        'debug': 0,
        'info': 1,
        'warn': 2,
        'error': 3,
        'off': 4,
    }
    _colors = {
        'debug': 'white',
        'info': 'blue',
        'warn': 'yellow',
        'error': 'red',
    }
    _signs = {
        'debug': '·',
        'info': '-',
        'warn': '!',
        'error': '‼',
    }
    _spinner = itertools.cycle('⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏')
    _spinner_done = '⣿'

    def __init__(self):
        super().__init__()
        self._level = self._levels['info']
        self.pause_level = self._levels['error']
        self.color = 'color' in os.environ['TERM']
        self.width = 80
        self._last_is_update = False
        self._last_level = self._level

    @property
    def width(self):
        width, _ = shutil.get_terminal_size((self._width, 24))
        return width

    @width.setter
    def width(self, value):
        self._width = value

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
        prev_level = self._level
        self._level = self._levels[level]
        yield
        self._level = prev_level

    @contextmanager
    def force_info_as_debug(self):
        _info = self.info
        self.info = self.debug
        yield
        self.info = _info

    def _header(self, text, level, update):
        if update:
            sign = next(self._spinner)
        else:
            sign = self._signs[level]
        return '{} {}'.format(colored(sign, self._colors[level]), text)

    def log(self, text, level='info', update=False):
        num_level = self._levels[level]
        if self._level > num_level:
            return
        if update:
            begin = '\r'
            end = ''
            header_len = 4
            width = self.width - header_len
            text += ' ' * width
            text = text[:width]
        else:
            begin = ''
            end = '\n'
        text = self._header(text, level, update)
        if not update and self._last_is_update:
            tick = colored(self._spinner_done, self._colors[self._last_level])
            begin = '\r{}\n{}'.format(tick, begin)
        print(begin + text, end=end)
        self._last_is_update = update
        self._last_level = level
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
