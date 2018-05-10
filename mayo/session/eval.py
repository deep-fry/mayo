import math

from mayo.log import log
from mayo.util import Percent, Table
from mayo.session.base import SessionBase


class Evaluate(SessionBase):
    mode = 'validate'

    def _finalize(self):
        super()._finalize()
        self.task.map_eval()

    def eval(self, key=None, keyboard_interrupt=True):
        # load checkpoint
        if key is None:
            key = self.config.system.checkpoint.load
        self.load_checkpoint(key)
        self.run(self.imgs_seen.initializer)
        # evaluation
        log.info('Starting evaluation...')
        num_iterations = math.ceil(self.num_examples / self.batch_size)
        try:
            for step in range(num_iterations):
                self.run([], batch=True)
        except KeyboardInterrupt as e:
            log.info('Evaluation aborted.')
            if not keyboard_interrupt:
                raise e
        else:
            log.info('Evaluation complete.')
        stats = {}
        # FIXME generalize this to detectors
        for name in ('top1', 'top5'):
            topn = []
            for each in self.estimator.get_history(name):
                topn += each.tolist()
            topn = topn[:self.num_examples]
            stats[name] = Percent(sum(topn) / len(topn))
            self.estimator.flush(name)
        log.info('    top1: {}, top5: {} [{} images]'.format(
            stats['top1'], stats['top5'], self.num_examples))
        return stats

    def _range(self, epochs):
        eval_range = self.config.get('eval.range', {})
        from_epoch = eval_range.get('from', 0)
        to_epoch = eval_range.get('to', -1)
        step = eval_range.get('step', 1)
        for e in epochs[::step]:
            if e < from_epoch:
                continue
            if to_epoch > 0 and e > to_epoch:
                continue
            yield e

    def eval_all(self):
        log.info('Evaluating all checkpoints...')
        epochs = list(self._range(self.checkpoint.list_epochs()))
        epochs_to_eval = ', '.join(str(e) for e in epochs)
        log.info('Checkpoints to evaluate: {}'.format(epochs_to_eval))
        results = Table(('Epoch', 'Top 1', 'Top 5'))
        # ensures imgs_seen initialized and loaded
        try:
            for e in epochs:
                with log.demote():
                    stats = self.eval(e, keyboard_interrupt=False)
                row = [e, stats['top1'], stats['top5']]
                results.add_row(row)
                log.info('epoch: {}, top1: {}, top5: {}'.format(*row))
        except KeyboardInterrupt:
            pass
        return results
