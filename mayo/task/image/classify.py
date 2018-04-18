import functools

import tensorflow as tf
from tensorflow.contrib import slim

from mayo.util import Percent, memoize_method
from mayo.task.image.base import ImageTaskBase


class Classify(ImageTaskBase):
    def __init__(
            self, session, preprocess,
            background_class, num_classes, shape, moment=None):
        self.label_offset = \
            int(background_class.get('use')) - \
            int(background_class.get('has'))
        session.config.dataset.task.num_classes += self.label_offset
        super().__init__(session, preprocess, shape, moment=None)

    def preprocess(self):
        for images, labels in super().preprocess():
            yield images, labels + self.label_offset

    @memoize_method
    def _train_setup(self, prediction, truth):
        # formatters
        accuracy_formatter = lambda e: \
            'accuracy: {}'.format(Percent(e.get_mean('accuracy')))
        # register progress update statistics
        accuracy = self._accuracy(prediction, truth)
        self.estimator.register(
            accuracy, 'accuracy', formatter=accuracy_formatter)

    def train(self, net, prediction, truth):
        self._train_setup(prediction, truth)
        truth = slim.one_hot_encoding(truth, prediction.shape[1])
        return tf.losses.softmax_cross_entropy(
            logits=prediction, onehot_labels=truth)

    @memoize_method
    def _eval(self):
        def top(prediction, truth, num_tops=1):
            return tf.nn.in_top_k(prediction, truth, num_tops)

        def metrics(net, prediction, truth):
            prediction = prediction['output']
            top1 = top(prediction, truth, 1)
            top5 = top(prediction, truth, 5)
            return top1, top5

        top1s, top5s = zip(*self.map(metrics))
        top1s = tf.concat(top1s, axis=0)
        top5s = tf.concat(top5s, axis=0)

        formatted_history = {}

        def formatter(estimator, name):
            history = formatted_history.setdefault(name, [])
            value = estimator.get_value(name)
            history.append(sum(value))
            accuracy = Percent(
                sum(history) / (self.session.batch_size * len(history)))
            return '{}: {}'.format(name, accuracy)

        for tensor, name in ((top1s, 'top1'), (top5s, 'top5')):
            self.estimator.register(
                tensor, name, history='infinite',
                formatter=functools.partial(formatter, name=name))

    def eval(self, net, prediction, truth):
        # set up eval estimators, once and for all predictions and truths
        return self._eval()
