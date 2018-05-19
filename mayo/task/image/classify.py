import os
import functools

import yaml
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from mayo.log import log
from mayo.util import Percent, memoize_method
from mayo.task.image.base import ImageTaskBase


class Classify(ImageTaskBase):
    _truth_keys = ['class/label']

    def transform(self, net, data, prediction, truth):
        truth = truth[0] + self.label_offset
        return data['input'], prediction['output'], truth

    @staticmethod
    def _warn_ties(ties, num_ties, thresholds):
        iterer = enumerate(zip(ties, num_ties, thresholds))
        for i, (each_ties, each_num_ties, each_threshold) in iterer:
            if each_num_ties == 1:
                continue
            indices = np.nonzero(each_ties)
            log.warn(
                'Top-k of batch index {} has {} tie values {} at indices {}.'
                .format(i, each_num_ties, each_threshold, indices))
        return num_ties

    def _top(self, prediction, truth, num_tops=1):
        # a full sort using top_k
        values, indices = tf.nn.top_k(prediction, self.num_classes)
        # cut-off threshold
        thresholds = values[:, (num_tops - 1):num_tops]
        # if > threshold, weight = 1, else weight = 0
        valids = tf.cast(prediction > thresholds, tf.float32)
        # ties should have weight = 1 / num_ties
        ties = tf.equal(prediction, thresholds)
        num_ties = tf.reduce_sum(
            tf.cast(ties, tf.float32), axis=-1, keepdims=True)
        num_ties = tf.py_func(
            self._warn_ties, [ties, num_ties, thresholds],
            tf.float32, stateful=False)
        num_ties = tf.tile(num_ties, [1, self.num_classes])
        weights = tf.where(ties, 1 / num_ties, valids)
        return slim.one_hot_encoding(truth, self.num_classes) * weights

    def _accuracy(self, prediction, truth, num_tops=1):
        top = self._top(prediction, truth, num_tops)
        return tf.reduce_sum(top) / top.shape.num_elements()

    @memoize_method
    def _train_setup(self, prediction, truth):
        # formatters
        accuracy_formatter = lambda e: \
            'accuracy: {}'.format(Percent(e.get_mean('accuracy', 'train')))
        # register progress update statistics
        accuracy = self._accuracy(prediction, truth)
        self.estimator.register(
            accuracy, 'accuracy', 'train', formatter=accuracy_formatter)

    def train(self, net, prediction, truth):
        self._train_setup(prediction, truth)
        truth = slim.one_hot_encoding(truth, self.num_classes)
        return tf.losses.softmax_cross_entropy(
            logits=prediction, onehot_labels=truth)

    @memoize_method
    def _eval_setup(self):
        def metrics(net, prediction, truth):
            top1 = self._top(prediction, truth, 1)
            top5 = self._top(prediction, truth, 5)
            return top1, top5

        top1s, top5s = zip(*self.map(metrics))
        top1s = tf.concat(top1s, axis=0)
        top5s = tf.concat(top5s, axis=0)

        formatted_history = {}

        def formatter(estimator, name):
            history = formatted_history.setdefault(name, [])
            value = estimator.get_value(name, 'eval')
            value = np.sum(value, axis=-1)
            history.append(sum(value))
            accuracy = Percent(
                sum(history) / (self.session.batch_size * len(history)))
            return '{}: {}'.format(name, accuracy)

        for tensor, name in ((top1s, 'top1'), (top5s, 'top5')):
            self.estimator.register(
                tensor, name, 'eval', history='infinite',
                formatter=functools.partial(formatter, name=name))

    def eval(self, net, prediction, truth):
        # set up eval estimators, once and for all predictions and truths
        return self._eval_setup()

    def eval_final_stats(self):
        stats = {}
        num_examples = self.session.num_examples
        num_remaining = num_examples % self.session.batch_size
        for key in ('top1', 'top5'):
            history = self.estimator.get_history(key, 'eval')
            history[-1] = history[-1][:num_remaining]
            valids = total = 0
            for h in history:
                valids += np.sum(h)
                total += len(h)
            stats[key] = Percent(valids / total)
            self.estimator.flush(key, 'eval')
        log.info(
            '    top1: {}, top5: {} [{} images]'
            .format(stats['top1'], stats['top5'], num_examples))

    def test(self, names, inputs, predictions):
        results = {}
        for name, image, prediction in zip(names, inputs, predictions):
            name = name.decode()
            label = self.class_names[np.argmax(prediction)]
            log.info('{} labeled as {}.'.format(name, label))
            results[name] = label
        output_dir = self.config.system.search_path.run.outputs[0]
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, 'predictions.yaml')
        with open(filename, 'w') as f:
            yaml.dump(results, f)
