import builtins


class MayoError(Exception):
    """The all-encompassing Mayo exception.  """


class NotImplementedError(MayoError, NotImplementedError):
    """Missing implementation.  """


class TypeError(MayoError, builtins.TypeError):
    """Incorrect type of object.  """


class ValueError(MayoError, builtins.ValueError):
    """Incorrect value used.  """


class KeyError(MayoError, builtins.KeyError):
    """Key not found.  """


class ShapeError(ValueError):
    """Incorrect shape.  """


class ConfigError(MayoError):
    """Incorrect configuration used.  """
