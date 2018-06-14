import re

from mayo.log import log
from mayo.util import memoize_property
from mayo.session.train import Train


class Search(Train):
    def _init_targets(self):
        # intialize target hyperparameter variables to search
        targets = {}
        for regex, info in self.config.search.variables.items():
            for node, node_variables in self.variables.items():
                for name, var in node_variables.items():
                    if not re.search(regex, name):
                        continue
                    if node in targets:
                        raise ValueError(
                            'We are currently unable to handle multiple '
                            'hyperparameter variables within the same layer.')
                    targets[node] = dict(info, variable=var)
                    log.debug(
                        'Targeted hyperparameter {} in {}: {}.'
                        .format(var, node.formatted_name(), info))
        return targets

    def _init_search(self):
        self.targets = self._init_targets()
        self.backtrack_targets = None
        # initialize # hyperparameters to starting positions
        # FIXME how can we continue search?
        for _, info in self.targets:
            start = info['from']
            var = info['variable']
            # unable to use overrider-based hyperparameter assignment, but it
            # shouldn't be a problem
            self.assign(var, start)
        # save a starting checkpoint for backtracking
        self.save_checkpoint('backtrack')

    def _priority(self, blacklist=None):
        key = self.config.search.cost_key
        info = self.task.nets[0].estimate()
        priority = [
            (node, stats[key]) for node, stats in info.items()
            if not blacklist or node not in blacklist]
        return list(reversed(sorted(priority, key=lambda v: v[1])))

    def backtrack(self):
        if not self.backtrack_targets:
            return False
        self.targets = self.backtrack_targets
        self.load_checkpoint('backtrack')
        return True

    def fine_tune(self):
        self.estimator.flush('accuracy', 'train')
        self._train_op = True
        self.config.system.max_epochs = self.config.search.max_epochs.fine_tune
        self._iteration()
        return self.estimator.get_value('accuracy', 'train')

    def _step_forward(self, value, end, step, min_step):
        new_value = value + step
        if step > 0 and new_value > end or step < 0 and new_value < end:
            # step size is too large, half it
            new_step = step / 2
            if new_step < min_step:
                # cannot step further
                return False
            return self._step_forward(value, end, step / 2, min_step)
        return new_value

    def search(self):
        # profile training accuracy for a given number of epochs
        self._profile()
        # initialize search
        self._init_search()
        # main procedure
        max_steps = self.config.search.max_steps
        step = 0
        blacklist = set()
        while True:
            if max_steps and step > max_steps:
                break
            step += 1
            priority = self._priority(blacklist)
            if not priority:
                log.debug('All nodes blacklisted.')
                break
            node = priority[0]
            info = self.targets[node]
            log.debug('Prioritize layer {!r}.'.format(node.formatted_name()))
            value = self._step_forward(info['from'], info['to'], info['step'])
            if value is False:
                blacklist.add(node)
                continue
            var = info['variable']
            self.assign(var, value)
            info['from'] = value
            log.info('Updated hyperparameter {} ')
            # fine-tuning with updated hyperparameter
            self.fine_tune()
            import pdb; pdb.set_trace()
            # TODO continue here...
        log.info('Automated hyperparameter optimization done.')

    def _profile(self):
        baseline = self.config.search.accuracy.get('baseline')
        if baseline:
            return baseline
        self.reset_num_epochs()
        log.info('Profiling baseline accuracy...')
        total_accuracy = 0
        step = epoch = 0
        while epoch < self.config.search.profile_epochs:
            epoch = self.run([self.num_epochs], batch=True)
            total_accuracy += self.estimator.get_value('accuracy', 'train')
            step += 1
        self.baseline = total_accuracy / step
        tolerance = self.config.search.accuracy.tolerance
        self.tolerable_baseline = self.baseline * tolerance
        log.info(
            'Baseline accuracy: {}, tolerable accuracy: .'
            .format(self.baseline, self.tolerable_baseline))
        self.reset_num_epochs()


class GlobalSearch(object):
    pass


class LayerwiseSearch(object):
    pass
