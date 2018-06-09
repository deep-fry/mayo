import os
import pickle
import tensorflow as tf

from mayo.log import log
from mayo.session.train import Train


class Profile(Train):
    def __init__(self, config):
        super().__init__(config)
        self.net = self.task.nets[0]
        self._run_train_ops = False
        self.config.system.max_epochs = 1

    def profile(self, reset=True):
        log.info('Start profiling ...')
        self.config.system.checkpoint.save = False
        # reset num_epochs and stop at 1 epoch
        if reset:
            self.reset_num_epochs()
        # start training
        self.train()
        # save profiling results
        self.info()
        self.save()

    def info(self):
        pass

    def save(self):
        path = self.config.system.search_path.profile[0]
        os.makedirs(path, exist_ok=True)
        # save estimator
        with open(os.path.join(path, 'estimator.pkl'), 'wb') as f:
            pickle.dump(self.estimator, f)
        # save variables
        variables = self.run(self.net.variables)
        with open(os.path.join(path, 'variables.pkl'), 'wb') as f:
            pickle.dump(variables, f)
