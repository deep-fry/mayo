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
            result = func(*args, **kwargs)
            setattr(self, name, result)
            return result
    return wrapped


@functools.lru_cache(maxsize=None)
def import_from_path(name, path):
    """
    Import module from path
    """
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_from_dot_path(path):
    components = path.split('.')
    m = __import__(components[0])
    for c in components[1:]:
        m = getattr(m, c)
    return m
