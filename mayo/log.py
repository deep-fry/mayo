import os
import sys
import time
import shutil
import itertools
from contextlib import contextmanager

from termcolor import colored


class Logger(object):
    _levels = {
        'debug': 0,
        'info': 1,
        'key': 2,
        'warn': 3,
        'error': 4,
        'off': 5,
    }
    _colors = {
        'debug': 'white',
        'info': 'blue',
        'key': 'green',
        'warn': 'yellow',
        'error': 'red',
    }
    _signs = {
        'debug': '·',
        'info': '-',
        'key': '*',
        'warn': '!',
        'error': '‼',
    }
    _spinner = itertools.cycle('⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏')
    _spinner_done = '⣿'

    def __init__(self):
        super().__init__()
        self._level = self._levels['info']
        self._pause_level = self._levels['error']
        self.color = 'color' in os.environ['TERM']
        self.width = 80
        self._last_is_update = False
        self._last_use_spinner = True
        self._last_level = self._level

    @property
    def width(self):
        width, _ = shutil.get_terminal_size((self._width, 24))
        return width

    @width.setter
    def width(self, value):
        self._width = value

    @classmethod
    def _level_key(cls, level):
        for k, v in cls._levels.items():
            if v == level:
                return k
        raise ValueError('Unrecognized log level.')

    @property
    def level(self):
        return self._level_key(self._level)

    @level.setter
    def level(self, value):
        self._level = self._levels[value]
        self.debug('Log level: {}'.format(value))

    @contextmanager
    def use_level(self, level):
        prev_level = self.level
        self.level = level
        yield
        self.level = prev_level

    @property
    def pause_level(self):
        return self._level_key(self._pause_level)

    @pause_level.setter
    def pause_level(self, value):
        self._pause_level = self._levels[value]
        self.debug('Log pause level: {}'.format(value))

    @contextmanager
    def use_pause_level(self, level):
        prev_level = self.pause_level
        self.pause_level = level
        yield
        self.pause_level = prev_level

    @contextmanager
    def demote(self):
        _key = self.key
        _info = self.info
        self.key = _info
        self.info = self.debug
        yield
        self.key = _key
        self.info = _info

    def _header(self, text, level, spinner):
        if spinner:
            sign = next(self._spinner)
        else:
            sign = self._signs[level]
        return '{} {}'.format(colored(sign, self._colors[level]), text)

    def log(self, text, level='info', update=False, spinner=True):
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
        text = self._header(text, level, update and spinner)
        if not update and self._last_is_update:
            if self._last_use_spinner:
                tick = colored(
                    self._spinner_done, self._colors[self._last_level])
                begin = '\r{}\n{}'.format(tick, begin)
            else:
                begin = '\n{}'.format(begin)
        print(begin + text, end=end)
        self._last_is_update = update
        self._last_use_spinner = update and spinner
        self._last_level = level
        while num_level >= self._pause_level:
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

    def debug(self, text, update=False, spinner=True):
        return self.log(text, 'debug', update, spinner)

    def info(self, text, update=False, spinner=True):
        return self.log(text, 'info', update, spinner)

    def key(self, text, update=False, spinner=True):
        return self.log(text, 'key', update, spinner)

    def warn(self, text, update=False, spinner=True):
        return self.log(text, 'warn', update, spinner)

    def error(self, text, update=False, spinner=True):
        return self.log(text, 'error', update, spinner)

    def countdown(self, text, secs, level='info'):
        try:
            for i in range(secs):
                msg = '{} in {} seconds... (Abort: ctrl+c)'
                msg = msg.format(text, secs - i)
                self.log(msg, level, update=True, spinner=False)
                time.sleep(1)
            return True
        except KeyboardInterrupt:
            log.info('We give up.')
            return False


log = Logger()
