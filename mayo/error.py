class MayoError(Exception):
    """The all-encompassing Mayo exception.  """


class NotImplementedError(MayoError, NotImplementedError):
    """Missing implementation.  """


class ShapeError(MayoError):
    """Incorrect shape.  """


class ConfigError(MayoError):
    """Incorrect configuration used.  """
