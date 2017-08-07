import yaml


class DotDict(dict):
    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        if isinstance(value, dict):
            return DotDict(value)
        return value

    def __getattr__(self, attr):
        return self[attr]
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Config(DotDict):
    def __init__(self, config_yaml):
        super().__init__(yaml.load(config_yaml))

    @property
    def input_shape(self):
        params = self.dataset
        return (params.height, params.width, params.channels)
