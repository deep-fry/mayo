import os
import sys
import time
import atexit
import shutil
import inspect
import itertools
from contextlib import contextmanager

from termcolor import colored

if os.name == "nt":
    import colorama
    colorama.init()

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

    _default_width = 80

    def __init__(self):
        super().__init__()
        self.level = 'info'
        self.pause_level = 'error'
        self.frame = False
        if 'TERM' in os.environ:
            self.color = 'color' in os.environ['TERM']
        else:
            self.color = True
        self._last_is_update = False
        self._last_use_spinner = True
        self._last_level = self.level

    @property
    def width(self):
        try:
            return self._width
        except AttributeError:
            pass
        width, _ = shutil.get_terminal_size((self._default_width, 24))
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

    def is_enabled(self, level):
        return self._level <= self._levels[level]

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

    def colored(self, text, level):
        return colored(text, self._colors[level])

    def debug_colored(self, text):
        return self.colored(text, 'debug')

    def info_colored(self, text):
        return self.colored(text, 'info')

    def key_colored(self, text):
        return self.colored(text, 'key')

    def warn_colored(self, text):
        return self.colored(text, 'warn')

    def error_colored(self, text):
        return self.colored(text, 'error')

    def _frame_info(self):
        # facepalm
        frame = inspect.currentframe().f_back.f_back.f_back.f_back
        file_name = frame.f_code.co_filename
        file_name = os.path.split(file_name)[1]
        file_name = os.path.splitext(file_name)[0]
        func_name = frame.f_code.co_name
        line_no = frame.f_lineno
        return '{}:{}#{}'.format(file_name, func_name, line_no)

    def _header(self, text, level, spinner):
        if spinner:
            sign = next(self._spinner)
        else:
            sign = self._signs[level]
        if self.frame:
            sign = self._frame_info()
        return '{} {}'.format(self.colored(sign, level), text)

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
                tick = self.colored(self._spinner_done, self._last_level)
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

    def error_exit(self, error_msg):
        with self.use_pause_level('off'):
            self.error(error_msg)
        sys.exit(-1)

    def countdown(self, text, secs, level='info'):
        try:
            for i in range(secs):
                msg = '{} in {} seconds... (Abort: ctrl+c)'
                msg = msg.format(text, secs - i)
                self.log(msg, level, update=True, spinner=False)
                time.sleep(1)
            return True
        except KeyboardInterrupt:
            log.debug('We give up.')
            return False

    def exit(self):
        # emit an empty line, as last log has no carriage return
        if self._last_is_update:
            print()


log = Logger()
atexit.register(log.exit)
