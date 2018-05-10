import numpy as np


class Change(object):
    def __init__(self, metric_count=100):
        super().__init__()
        self._persistence = {}
        self._metric_count = metric_count

    def delta(self, name, value):
        name += '.delta'
        prev_value = self._persistence.get(name, value)
        self._persistence[name] = value
        return value - prev_value

    def every(self, name, value, interval):
        if interval <= 0:
            return False
        name += '.every'
        next_value = self._persistence.setdefault(name, value) + interval
        if value < next_value:
            return False
        self._persistence[name] = value
        return True

    def moving_metrics(self, name, value, std=True, over=None):
        name += '.moving'
        history = self._persistence.setdefault(name, [])
        over = over or self._metric_count
        while len(history) >= over:
            history.pop(0)
        history.append(value)
        mean = np.mean(history)
        if not std:
            return mean
        return mean, np.std(history)

    def reset(self, name):
        for key in list(self._persistence):
            if not key.startswith(name + '.'):
                continue
            del self._persistence[key]
