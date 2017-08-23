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


def import_from_string(string):
    if ':' in string:
        path, dot_path = string.split(':')
        mod = import_from_file(path)
    else:
        mod = None
    return import_from_dot_path(string, mod)
