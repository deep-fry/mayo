import os
import pickle
import tensorflow as tf

from mayo.log import log
from mayo.session.train import Train


class Profile(Train):
    def __init__(self, config):
        super().__init__(config)
        self.net = self.task.nets[0]

    def profile(self, overriders=None):
        log.info('Start profiling for one epoch...')
        if self.config.system.profile.activations:
            self._register_activations()
        if overriders:
            self._register_quantize_loss(overriders)
        # disable checkpoint saving and train_op
        self.config.system.checkpoint.save = False
        self._run_train_ops = False
        # reset num_epochs and stop at 1 epoch
        self.reset_num_epochs()
        self.config.system.max_epochs = 1
        # start training
        self.train()
        # save profiling results
        self.info()
        self.save()

    def _register_quantize_loss(self, overriders):
        for o in overriders:
            loss = o.quantize_loss()
            self.estimator.register(
                loss, 'activation' + o.name)

    def _register_activations(self):
        history = self.config.dataset.num_examples_per_epoch.get(self.mode)
        for node, tensor in self.net.layers().items():
            # FIXME
            # store max values only for now
            # self.estimator.register(
            #     tensor, 'activation', node=node, history=history)
            # store the topk
            # values, indices = tf.nn.top_k(tf.abs(tensor), k=10)
            # tensor = tf.cast(tensor, tf.float32)
            values = tf.reduce_max(tf.abs(tensor))
            self.estimator.register(
                values, 'activation', node=node, history=history)

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
