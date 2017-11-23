import math

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
        self._top1_op = tf.concat(top1s, axis=0)
        self._top5_op = tf.concat(top5s, axis=0)

    def eval(self, key=None, keyboard_interrupt=True):
        # load checkpoint
        if key is None:
            key = self.config.system.checkpoint.load
        self.load_checkpoint(key)
        num_examples = self.config.dataset.num_examples_per_epoch.validate
        num_iterations = math.ceil(num_examples / self.batch_size)
        num_final_examples = num_examples % self.batch_size
        # evaluation
        log.info('Starting evaluation...')
        top1s, top5s, step, total = 0.0, 0.0, 0, 0
        try:
            while step < num_iterations:
                top1, top5 = self.run(
                    [self._top1_op, self._top5_op], update_progress=True)
                if step == num_iterations - 1:
                    # final iteration
                    top1 = top1[:num_final_examples]
                    top5 = top5[:num_final_examples]
                    total += num_final_examples
                else:
                    total += self.batch_size
                top1s += sum(top1)
                top5s += sum(top5)
                top1_acc = Percent(top1s / total)
                top5_acc = Percent(top5s / total)
                self.register_update('eval', Percent(total / num_examples))
                self.register_update('top1', top1_acc)
                self.register_update('top5', top5_acc)
                self._register_overrider('mayo.overrider.gates')
                step += 1
        except KeyboardInterrupt as e:
            log.info('Evaluation aborted.')
            if not keyboard_interrupt:
                raise e
        else:
            log.info('Evaluation complete.')
            log.info('    top1: {}, top5: {} [{} images]'.format(
                top1_acc, top5_acc, total))
            return top1_acc, top5_acc

    def _register_overrider(self, collection):
        # a hack
        gates = tf.get_collection(collection)
        if not gates:
            return
        valid = tf.add_n([tf.reduce_sum(g) for g in gates])
        total = sum(g.shape.num_elements() for g in gates)
        density = valid / total
        density_formatter = lambda d: Percent(
            self.change.moving_metrics('density', d, std=False))
        self.register_update('density', density, density_formatter)

    def eval_all(self):
        log.info('Evaluating all checkpoints...')
        epochs = self.checkpoint.list_epochs()
        epochs_to_eval = ', '.join(str(e) for e in epochs)
        log.info('Checkpoints to evaluate: {}'.format(epochs_to_eval))
        results = Table(('Epoch', 'Top 1', 'Top 5'))
        # ensures imgs_seen initialized and loaded
        epochs_op = self.num_epochs
        try:
            for e in epochs:
                with log.demote():
                    top1, top5 = self.eval(e, keyboard_interrupt=False)
                top1, top5 = Percent(top1), Percent(top5)
                row = ('{:.3f}'.format(self.run(epochs_op)), top1, top5)
                results.add_row(row)
                log.info('epoch: {}, top1: {}, top5: {}'.format(*row))
        except KeyboardInterrupt:
            pass
        return results.format()


class Evaluate(EvaluateBase):
    num_gpus = 1


class FastEvaluate(EvaluateBase):
    pass
