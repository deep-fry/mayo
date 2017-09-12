import time
import math

import tensorflow as tf

from mayo.log import log
from mayo.net import Net
from mayo.checkpoint import CheckpointHandler
from mayo.preprocess import Preprocess
from mayo.train import _global_step, _imgs_seen
from mayo.util import format_percent, tabular


class Evaluate(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._graph = tf.Graph()
        self._session = tf.Session(graph=self._graph)
        self._preprocessor = Preprocess(self.config)
        system = self.config.system
        self._checkpoint = CheckpointHandler(
            self._session, system.checkpoint.load, system.checkpoint.save,
            system.search_paths.checkpoints)
        with self._graph.as_default():
            self._init()

    def _init(self):
        log.info('Initializing...')
        # network
        images, labels = self._preprocessor.preprocess_validate()
        self._net = Net(self.config, images, labels, False, self._graph)
        logits = self._net.logits()
        # moving average decay
        decay = self.config.get('train.moving_average_decay', None)
        if decay:
            log.debug('Using exponential moving average.')
            var_avgs = tf.train.ExponentialMovingAverage(decay, _global_step())
            var_avgs.apply(
                tf.trainable_variables() + tf.moving_average_variables())
        else:
            log.debug('Not using exponential moving average.')
        # metrics
        self._top1_op = tf.nn.in_top_k(logits, labels, 1)
        self._top5_op = tf.nn.in_top_k(logits, labels, 5)
        # initialization
        self._session.run(tf.global_variables_initializer())
        # queue runners
        tf.train.start_queue_runners(sess=self._session)

    def _update_progress(self, step, top1, top5, num_iterations):
        now = time.time()
        duration = now - getattr(self, '_prev_time', now)
        if duration != 0:
            num_steps = step - getattr(self, '_prev_step', step)
            imgs_per_sec = self.config.system.batch_size * num_steps
            imgs_per_sec /= float(duration)
            percentage = step / num_iterations * 100
            info = 'eval: {:.2f}% | top1: {:.2f}% | top5: {:.2f}% | {:.1f}/s'
            info = info.format(
                percentage, top1 * 100, top5 * 100, imgs_per_sec)
            log.info(info, update=True)
        self._prev_time = now
        self._prev_step = step

    def _eval(self, epoch=None, keyboard_interrupt=True):
        # load checkpoint
        self._checkpoint.load(epoch)
        num_examples = self.config.dataset.num_examples_per_epoch.validate
        batch_size = self.config.system.batch_size
        num_iterations = math.ceil(num_examples / batch_size)
        num_final_examples = num_examples % batch_size
        # evaluation
        log.info('Starting evaluation...')
        top1s, top5s, step, total = 0.0, 0.0, 0, 0
        try:
            while step < num_iterations:
                top1, top5 = self._session.run([self._top1_op, self._top5_op])
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
            log.info('\ttop1: {}, top5: {} [{} images]'.format(
                format_percent(top1_acc), format_percent(top5_acc), total))
        return top1_acc, top5_acc

    def eval(self):
        with self._graph.as_default():
            return self._eval()

    def eval_all(self):
        log.info('Evaluating all checkpoints...')
        checkpoints = self._checkpoint.list()
        log.debug('Checkpoints to evaluate:')
        for c in checkpoints:
            log.debug('    {}'.format(c))
        imgs_per_epoch = self.config.dataset.num_examples_per_epoch.train
        results = []
        with self._graph.as_default():
            imgs_seen = _imgs_seen()
            try:
                for c in checkpoints:
                    with log.force_info_to_debug():
                        top1, top5 = self._eval(c, keyboard_interrupt=False)
                    epoch = self._session.run(imgs_seen) / imgs_per_epoch
                    epoch_str = '{:.3f}'.format(epoch)
                    top1 = format_percent(top1)
                    top5 = format_percent(top5)
                    log.info('epoch: {}, top1: {}, top5: {}'.format(
                        epoch_str, top1, top5))
                    results.append((epoch, epoch_str, top1, top5))
            except KeyboardInterrupt:
                pass
        table = [('Epoch', 'Top 1', 'Top 5'), '-']
        for epoch, *result in sorted(results):
            table.append(result)
        return tabular(table)
