import time
import math

import tensorflow as tf

from mayo.log import log
from mayo.net import Net
from mayo.util import delta, moving_metrics, format_percent, tabular
from mayo.session import Session


class Evaluate(Session):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        with self.as_default():
            self._init()

    def _init(self):
        log.info('Instantiating...')
        # network
        images, labels = self.preprocessor.preprocess_validate()
        self._net = Net(self.config, images, labels, False)
        logits = self._net.logits()
        # moving average decay
        avg_op = self.moving_average_op()
        log.debug(
            ('Using' if avg_op else 'Not using') +
            ' exponential moving average.')
        # metrics
        self._top1_op = tf.nn.in_top_k(logits, labels, 1)
        self._top5_op = tf.nn.in_top_k(logits, labels, 5)
        # initialization
        self.init()

    def _update_progress(self, step, top1, top5, num_iterations):
        interval = delta('eval.duration', time.time())
        if interval != 0:
            batch_size = self.config.system.batch_size
            metric_count = self.config.system.metrics_history_count
            imgs_per_sec = batch_size * delta('eval.step', step) / interval
            imgs_per_sec = moving_metrics(
                'eval.imgs_per_sec', imgs_per_sec,
                std=False, over=metric_count)
            percentage = step / num_iterations * 100
            info = 'eval: {:.2f}% | top1: {:.2f}% | top5: {:.2f}% | {:.1f}/s'
            info = info.format(
                percentage, top1 * 100, top5 * 100, imgs_per_sec)
            log.info(info, update=True)

    def eval(self, epoch=None, keyboard_interrupt=True):
        # load checkpoint
        self.checkpoint.load(epoch)
        num_examples = self.config.dataset.num_examples_per_epoch.validate
        batch_size = self.config.system.batch_size
        num_iterations = math.ceil(num_examples / batch_size)
        num_final_examples = num_examples % batch_size
        # evaluation
        log.info('Starting evaluation...')
        top1s, top5s, step, total = 0.0, 0.0, 0, 0
        try:
            while step < num_iterations:
                top1, top5 = self.run([self._top1_op, self._top5_op])
                if step == num_iterations - 1:
                    # final iteration
                    top1 = top1[:num_final_examples]
                    top5 = top5[:num_final_examples]
                    total += num_final_examples
                else:
                    total += batch_size
                top1s += sum(top1)
                top5s += sum(top5)
                top1_acc = top1s / total
                top5_acc = top5s / total
                step += 1
                self._update_progress(step, top1_acc, top5_acc, num_iterations)
        except KeyboardInterrupt as e:
            log.info('Evaluation aborted.')
            if not keyboard_interrupt:
                raise e
        else:
            log.info('Evaluation complete.')
            log.info('    top1: {}, top5: {} [{} images]'.format(
                format_percent(top1_acc), format_percent(top5_acc), total))
        return top1_acc, top5_acc

    def eval_all(self):
        log.info('Evaluating all checkpoints...')
        epochs = self.checkpoint.list_epochs()
        epochs_to_eval = ', '.join(str(e) for e in epochs)
        log.info('Checkpoints to evaluate: {}'.format(epochs_to_eval))
        imgs_per_epoch = self.config.dataset.num_examples_per_epoch.train
        imgs_seen = self.imgs_seen
        results = []
        try:
            for e in epochs:
                with log.force_info_as_debug():
                    top1, top5 = self.eval(e, keyboard_interrupt=False)
                epoch = self.run(imgs_seen) / imgs_per_epoch
                epoch_str = '{:.3f}'.format(epoch)
                top1 = format_percent(top1)
                top5 = format_percent(top5)
                log.info('epoch: {}, top1: {}, top5: {}'.format(
                    epoch_str, top1, top5))
                results.append((epoch_str, top1, top5))
        except KeyboardInterrupt:
            pass
        return tabular(results)
