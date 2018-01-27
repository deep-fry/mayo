import math
import pickle
import os
from mayo.util import Percent
import tensorflow as tf

from mayo.log import log
from mayo.session.eval import EvaluateBase


class Profile_stats(EvaluateBase):
    def __init__(self, config):
        super().__init__(config)
        if config.get('profile') is None:
            raise ValueError('missing profile yaml')
        self.profile_config = config.profile
        self.mode = self.profile_config.get('mode', 'train')
        self._setup()
        self.net = self.nets[0]

    def profile(self, key=None, keyboard_interrupt=True):
        if key is None:
            key = self.config.system.checkpoint.load
        self.load_checkpoint(key)
        self.run(self.imgs_seen.initializer)
        log.info('Starts profiling in {} mode'.format(self.mode))
        num_iterations = math.ceil(self.num_examples / self.batch_size)
        # check saving dir
        save_dir = self.profile_config.get('save_dir', './profile_stats/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if self.profile_config.get('activations', False):
            history = self.profile_config.get('hisotry', 1000)
            self._profile_activations(save_dir, history)
        try:
            for step in range(num_iterations):
                self.run([], batch=True)
        except KeyboardInterrupt as e:
            log.info('Stats profiling aborted.')
            if not keyboard_interrupt:
                raise e
        else:
            log.info('Profiling complete.')
        stats = {}
        for name in ('top1', 'top5'):
            topn = []
            for each in self.estimator.get_history(name, 'global'):
                topn += each.tolist()
            topn = topn[:self.num_examples]
            stats[name] = Percent(sum(topn) / len(topn))
        log.info('    top1: {}, top5: {} [{} images]'.format(
            stats['top1'], stats['top5'], self.num_examples))
        if self.config.profile.get('activations', False):
            self._store_actiations(save_dir)
        if self.config.profile.get('weights', False):
            self._store_variables(save_dir)
        if self.config.profile.get('gate', False):
            self._store_gates(save_dir)
        return

    def _store_variables(self, directory):
        np_variables = self.run(self.net.variables)
        # replace keys so its not weak ref anymore
        for node in list(np_variables.keys()):
            np_variables[node.name] = np_variables.pop(node)
        with open(directory + 'variables.pkl', 'wb') as f:
            pickle.dump(np_variables, f)

    def _profile_activations(self, directory, history=1000):
        all_layers = {}
        for node, item in self.net.layers().items():
            all_layers[node.name] = item

        self.estimator.register(
            all_layers, 'activations', history=history)
        # moving_mean = 0.99 *
        # self.estimator.register(
        #     item, 'actvations_mean', node, history=1
        # )

    def _store_actiations(self, directory):
        # layers = {}
        # for node, item in self.net.layers().items():
        #     layers[node.name] = self.estimator.get_history(
        #         'activations', node)
        all_layers = self.estimator.get_history('activations')
        with open(directory + 'activations.pkl', 'wb') as f:
            pickle.dump(all_layers, f)

    def _store_gates(self, directory):
        # check whether gate density exisits
        gate_densities = {}
        for node, item in self.net.layers().items():
            try:
                mean_density = self.estimator.get_mean('gate', node)
                gate_densities[node.name] = mean_density
            except KeyError:
                log.info('Gate densities are not registered in statistics')
        if gate_densities == {}:
            return
        with open(directory + 'gate_densities.pkl', 'wb') as f:
            pickle.dump(gate_densities, f)
