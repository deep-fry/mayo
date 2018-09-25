import re
import math

import tensorflow as tf

from mayo.log import log
from mayo.util import memoize_property
from mayo.session.train import Train


class SearchBase(Train):
    def _profile(self):
        baseline = self.config.search.accuracy.get('baseline')
        if baseline:
            return baseline
        self.reset_num_epochs()
        log.info('Profiling baseline accuracy...')
        total_accuracy = step = epoch = 0
        while epoch < self.config.search.max_epochs.profile:
            epoch = self.run(self.num_epochs, batch=True)
            total_accuracy += self.estimator.get_value('accuracy', 'train')
            step += 1
        self.baseline = total_accuracy / step
        tolerance = self.config.search.accuracy.tolerance
        self.tolerable_baseline = self.baseline * (1 - tolerance)
        log.info(
            'Baseline accuracy: {}, tolerable accuracy: {}.'
            .format(self.baseline, self.tolerable_baseline))
        self.reset_num_epochs()

    def _init_targets(self):
        # intialize target hyperparameter variables to search
        targets = {}
        dtypes = {
            tf.int32: int,
            tf.int32._as_ref: int,
            tf.float32: float,
            tf.float32._as_ref: float,
        }
        for regex, info in self.config.search.variables.items():
            for node, node_variables in self.variables.items():
                node_name = node.formatted_name()
                for name, var in node_variables.items():
                    var_name = '{}/{}'.format(node_name, name)
                    if not re.search(regex, var_name):
                        continue
                    if node in targets:
                        raise ValueError(
                            'We are currently unable to handle multiple '
                            'hyperparameter variables within the same layer.')
                    try:
                        dtype = dtypes[var.dtype]
                    except KeyError:
                        raise TypeError(
                            'We accept only integer or floating-point '
                            'hyperparameters.')
                    targets[node] = dict(info, variable=var, type=dtype)
                    log.debug(
                        'Targeting hyperparameter {} in {}: {}...'
                        .format(var, node_name, targets[node]))
        return targets

    def _init_search(self):
        self.targets = self._init_targets()
        if not self.targets:
            raise ValueError(
                'No search target hyperparameter specified. '
                'Perhaps your regex is not written correctly?')
        # initialize hyperparameters to starting positions
        # FIXME how can we continue search?
        for _, info in self.targets.items():
            start = info['from']
            var = info['variable']
            # unable to use overrider-based hyperparameter assignment, but it
            # shouldn't be a problem
            self.assign(var, start)
        self.set_backtrack_to_here()

    def _reduce_step(self, step, dtype):
        if dtype is float:
            return step / 2
        if dtype is int:
            sign = 1 if step > 0 else -1
            step = abs(step)
            new_step = int(math.ceil(step / 2))
            if new_step == step:
                return sign * (step - 1)
            return sign * new_step
        raise TypeError('Unrecognized data type.')

    def _step_forward(self, value, end, step, min_step, dtype):
        new_value = value + step
        if step > 0 and new_value > end or step < 0 and new_value < end:
            # step size is too large, half it
            new_step = self._reduce_step(step, dtype)
            if abs(new_step) < abs(min_step):
                # cannot step further
                return False
            return self._step_forward(value, end, new_step, min_step, dtype)
        return new_value

    def backtrack(self):
        log.debug('Reverting to the last hyperparameters.')
        self.targets = self.backtrack_targets
        self.load_checkpoint('backtrack')

    def set_backtrack_to_here(self):
        self.backtrack_targets = {}
        for node, info in self.targets.items():
            self.backtrack_targets[node] = dict(info)
        self.save_checkpoint('backtrack')

    def fine_tune(self):
        self.reset_num_epochs()
        self.overriders_update()
        max_epoch = self.config.search.max_epochs.fine_tune
        total_accuracy = step = epoch = 0
        while epoch < max_epoch:
            epoch, _ = self.run([self.num_epochs, self._train_op], batch=True)
            total_accuracy += self.estimator.get_value('accuracy', 'train')
            step += 1
        accuracy = total_accuracy / step
        if accuracy >= self.tolerable_baseline:
            log.debug(
                'Fine-tuned accuracy {!r} found tolerable.'
                .format(accuracy))
            self.set_backtrack_to_here()
            return True
        return False

    def kernel(self):
        raise NotImplementedError

    def search(self):
        # profile training accuracy for a given number of epochs
        self._profile()
        # initialize search
        self._init_search()
        # main procedure
        max_steps = self.config.search.max_steps
        step = 0
        while True:
            if max_steps and step > max_steps:
                break
            step += 1
            if not self.kernel():
                break
        log.info('Automated hyperparameter optimization complete.')


class Search(SearchBase):
    def _init_search(self):
        super()._init_search()
        self.blacklist = set()

    def _priority(self, blacklist=None):
        key = self.config.search.cost_key
        info = self.task.nets[0].estimate()
        priority = []
        for node, stats in info.items():
            if node not in self.targets:
                continue
            if blacklist and node in blacklist:
                continue
            priority.append((node, stats[key]))
        return list(reversed(sorted(priority, key=lambda v: v[1])))

    def kernel(self):
        mode = self.config.search.mode
        try:
            kernel_func = getattr(self, '{}_kernel'.format(mode))
        except AttributeError:
            raise AttributeError(
                'We cannot search with unrecognized mode {!r}.'.format(mode))
        return kernel_func()

    def priority_kernel(self):
        priority = self._priority(self.blacklist)
        if not priority:
            log.debug('All nodes blacklisted.')
            return False
        node, node_priority = priority[0]
        info = self.targets[node]
        node_name = node.formatted_name()
        log.debug(
            'Prioritize layer {!r} with importance {}.'
            .format(node_name, node_priority))
        value = self._step_forward(
            info['from'], info['to'], info['step'],
            info['min_step'], info['type'])
        var = info['variable']
        if value is False:
            log.debug(
                'Blacklisting {!r} as we cannot further '
                'increment/decrement {!r}.'.format(node_name, var))
            self.blacklist.add(node)
            return True
        self.assign(var, value)
        info['from'] = value
        log.info(
            'Updated hyperparameter {!r} in layer {!r} with a new value {}.'
            .format(var.op.name, node_name, value))
        # fine-tuning with updated hyperparameter
        tolerable = self.fine_tune()
        if tolerable:
            # satisfies the budget constraint
            return True
        # fail to satisfy, backtrack and decrement step size
        self.backtrack()
        info = self.targets[node]
        new_step = self._reduce_step(info['step'], info['type'])
        if new_step < info['min_step']:
            self.blacklist.add(node)
            log.debug(
                'Blacklisting {!r} as we cannot use smaller '
                'increment/decrement.'.format(node_name))
            return True
        info['step'] = new_step
        return True

    def global_kernel(self):
        for node, info in self.targets.items():
            node_name = node.formatted_name()
            value = self._step_forward(
                info['from'], info['to'], info['step'], info['min_step'],
                info['type'])
            var = info['variable']
            if value is False:
                log.debug(
                    'Stopping because of {!r}, as we cannot further '
                    'increment/decrement {!r}.'.format(node_name, var))
                return False
            self.assign(var, value)
            info['from'] = value
            log.info(
                'Updated hyperparameter {!r} in layer {!r} with a new '
                'value {}.'.format(var.op.name, node_name, value))
        # fine-tuning with updated hyperparameter
        tolerable = self.fine_tune()
        if tolerable:
            return True
        self.backtrack()
        for node, info in self.targets.items():
            new_step = self._reduce_step(info['step'], info['type'])
            if abs(new_step) < abs(info['min_step']):
                log.debug(
                    'Stopping because of {!r}, as we cannot use smaller '
                    'increment/decrement.'.format(node_name))
                return False
            info['step'] = new_step
        return True
