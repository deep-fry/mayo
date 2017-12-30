import math
import functools

import tensorflow as tf

from mayo.log import log
from mayo.util import Percent, Table
from mayo.session.base import Session


class EvaluateBase(Session):
    mode = 'validate'

    def __init__(self, config):
        super().__init__(config)
        self._setup()

    def _setup(self):
        # moving average decay
        avg_op = self.moving_average_op()
        using = 'Using' if avg_op else 'Not using'
        log.debug(using + ' exponential moving averages.')
        # setup metrics
        metrics_func = lambda net: (net.top(1), net.top(5))
        top1s, top5s = zip(*self.net_map(metrics_func))
        top1s = tf.concat(top1s, axis=0)
        top5s = tf.concat(top5s, axis=0)

        formatted_history = {}

        def formatter(estimator, name):
            history = formatted_history.setdefault(name, [])
            value = estimator.get_value(name)
            history.append(sum(value))
            accuracy = Percent(sum(history) / (self.batch_size * len(history)))
            return '{}: {}'.format(name, accuracy)

        self.estimator.register(
            top1s, 'top1', history='infinite',
            formatter=functools.partial(formatter, name='top1'))
        self.estimator.register(
            top5s, 'top5', history='infinite',
            formatter=functools.partial(formatter, name='top5'))

    def eval(self, key=None, keyboard_interrupt=True):
        # load checkpoint
        if key is None:
            key = self.config.system.checkpoint.load
        self.load_checkpoint(key)
        # evaluation
        log.info('Starting evaluation...')
        num_examples = self.config.dataset.num_examples_per_epoch.validate
        num_iterations = math.ceil(num_examples / self.batch_size)
        try:
            for step in range(num_iterations):
                self.run([], update_progress=True)
        except KeyboardInterrupt as e:
            log.info('Evaluation aborted.')
            if not keyboard_interrupt:
                raise e
        else:
            log.info('Evaluation complete.')
        stats = {}
        for name in ('top1', 'top5'):
            topn = []
            for each in self.estimator.get_history(name):
                topn += each.tolist()
            if len(topn) != num_iterations * self.batch_size:
                raise ValueError(
                    'Number of top statistics should match number of '
                    'iterations.')
            topn = topn[:num_examples]
            stats[name] = Percent(sum(topn) / len(topn))
        log.info('    top1: {}, top5: {} [{} images]'.format(
            stats['top1'], stats['top5'], num_examples))
        return stats

    def eval_all(self):
        log.info('Evaluating all checkpoints...')
        epochs = self.checkpoint.list_epochs()
        epochs_to_eval = ', '.join(str(e) for e in epochs)
        log.info('Checkpoints to evaluate: {}'.format(epochs_to_eval))
        results = Table(('Epoch', 'Top 1', 'Top 5'))
        interval = self.config.get('eval.interval', 1)
        # ensures imgs_seen initialized and loaded
        epochs_op = self.num_epochs
        try:
            for e in epochs[::interval]:
                with log.demote():
                    stats = self.eval(e, keyboard_interrupt=False)
                row = ['{:.3f}'.format(self.run(epochs_op))]
                row += [stats['top1'], stats['top5']]
                results.add_row(row)
                log.info('epoch: {}, top1: {}, top5: {}'.format(*row))
        except KeyboardInterrupt:
            pass
        return results


class Evaluate(EvaluateBase):
    num_gpus = 1


class FastEvaluate(EvaluateBase):
    pass
