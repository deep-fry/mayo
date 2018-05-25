import os
import collections
from contextlib import contextmanager

import tensorflow as tf

from mayo.log import log
from mayo.util import memoize_property
from mayo.net.tf import TFNet
from mayo.session.test import Test


class TFTaskBase(object):
    """Specifies common training and evaluation tasks.  """
    debug = False

    def __init__(self, session):
        super().__init__()
        self.is_test = isinstance(session, Test)
        self.session = session
        self.config = session.config
        self.num_gpus = self.config.system.num_gpus
        self.mode = session.mode
        self.estimator = session.estimator
        self._instantiate_nets()

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

    @staticmethod
    def _test_files(folder):
        suffixes = ['.jpg', '.jpeg', '.png']
        files = [
            name for name in sorted(os.listdir(folder))
            if any(name.endswith(s) for s in suffixes)]
        log.debug(
            'Running in folder {!r} on images: {}'
            .format(folder, ', '.join(files)))
        return [os.path.join(folder, name) for name in files]

    def _instantiate_nets(self):
        nets = []
        inputs = []
        predictions = []
        truths = []
        names = []
        model = self.config.model
        iterer = self.generate()
        for i, (data, additional) in enumerate(iterer):
            if self.is_test:
                name, truth = additional[0], None
            else:
                name, truth = None, additional
            log.debug('Instantiating graph for GPU #{}...'.format(i))
            with self._gpu_context(i):
                net = TFNet(self.session, model, data, bool(nets))
            nets.append(net)
            prediction = net.outputs()
            data, prediction, truth = self.transform(
                net, data, prediction, truth)
            if i == 0 and self.debug:
                self._register_estimates(prediction, truth)
            inputs.append(data)
            predictions.append(prediction)
            truths.append(truth)
            names.append(name)
        self.nets = nets
        self.inputs = inputs
        self.predictions = predictions
        self.truths = truths
        self.names = names

    def _register_estimates(self, prediction, truth):
        def register(root, mapping):
            history = 'infinite' if self.mode == 'validate' else None
            if not isinstance(mapping, collections.Mapping):
                if mapping is not None:
                    self.estimator.register(mapping, root, history=history)
                return
            for key, value in mapping.items():
                register('{}.{}'.format(root, key), value)
        register('prediction', prediction)
        register('truth', truth)

    def transform(self, net, data, prediction, truth):
        return data, prediction, truth

    def generate(self):
        raise NotImplementedError(
            'Please implement .generate() which produces training/validation '
            'samples and the expected truth results.')

    def train(self, net, prediction, truth):
        raise NotImplementedError(
            'Please implement .train() which returns the loss tensor.')

    def eval(self):
        raise NotImplementedError(
            'Please implement .eval() which registers the evaluation metrics.')

    def post_eval(self):
        raise NotImplementedError(
            'Please impelement .post_eval() which computes an info dict '
            'for the evaluation metrics.')

    def test(self, name, prediction):
        raise NotImplementedError(
            'Please implement .test() which produces human-readable output '
            'for a given input.')
