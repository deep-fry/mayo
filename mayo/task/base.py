from contextlib import contextmanager

import tensorflow as tf

from mayo.log import log
from mayo.error import NotImplementedError
from mayo.net.tf import TFNet


class TFTaskBase(object):
    """Specifies common training and evaluation tasks.  """
    def __init__(self, session):
        super().__init__()
        self.session = session
        self.config = session.config
        self.num_gpus = self.config.system.num_gpus
        self.mode = session.mode
        self.estimator = session.estimator
        self.nets, self.predictions, self.truths = self._instantiate_nets()
        self._register_estimates()

    @contextmanager
    def _gpu_context(self, gid):
        with tf.device('/gpu:{}'.format(gid)):
            with tf.name_scope('tower_{}'.format(gid)) as scope:
                yield scope

    def map(self, func):
        iterer = enumerate(zip(self.nets, self.predictions, self.truths))
        for i, (net, prediction, truth) in iterer:
            with self._gpu_context(i):
                yield func(net, prediction, truth)

    def _instantiate_nets(self):
        nets = []
        predictions = []
        truths = []
        model = self.config.model
        for i, (inputs, truth) in enumerate(self.preprocess()):
            log.debug('Instantiating graph for GPU #{}...'.format(i))
            with self._gpu_context(i):
                net = TFNet(self.session, model, inputs, bool(nets))
            nets.append(net)
            predictions.append(net.outputs())
            truths.append(truth)
        return nets, predictions, truths

    def _register_estimates(self):
        history = 'infinite' if self.mode == 'validate' else None
        for key, value in self.predictions[0].items():
            self.estimator.register(
                value, 'predictions.{}'.format(key), history=history)
        self.estimator.register(self.truths[0], 'truth', history=history)

    def preprocess(self):
        raise NotImplementedError(
            'Please implement .preprocess() which produces training samples '
            'and the expected truth results.')

    def train(self, net, prediction, truth):
        raise NotImplementedError(
            'Please implement .train() which returns the loss tensor.')

    def eval(self, net, prediction, truth):
        raise NotImplementedError(
            'Please implement .eval() which returns the evaluation metrics.')

    def map_train(self):
        return list(self.map(self.train))

    def map_eval(self):
        return list(self.map(self.eval))
