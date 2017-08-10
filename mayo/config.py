import yaml as yamllib


class _DotDict(dict):
    def __init__(self, data):
        super().__init__({})
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        if isinstance(value, dict):
            return _DotDict(value)
        return value

    def __getattr__(self, attr):
        return self[attr]
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Config(_DotDict):
    def __init__(self, yaml=None, path=None):
        if path:
            with open(path, 'r') as file:
                yaml = file.read()
        super().__init__(yamllib.load(yaml))

    @property
    def input_shape(self):
        params = self.dataset
        return (params.height, params.width, params.channels)
