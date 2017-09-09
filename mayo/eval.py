import time
import math

import numpy as np
import tensorflow as tf

from mayo.log import log
from mayo.net import Net
from mayo.checkpoint import CheckpointHandler
from mayo.preprocess import Preprocess
from mayo.train import _global_step


class Evaluate(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._graph = tf.Graph()
        self._preprocessor = Preprocess(self.config)

    def logits(self, images, labels, reuse):
        self._net = Net(
            self.config, images, labels, False, graph=self._graph, reuse=reuse)
        return self._net.logits()

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

    def _eval(self):
        # network
        images, labels = self._preprocessor.preprocess_validate()
        logits = self.logits(images, labels, False)

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
        top1_op = tf.nn.in_top_k(logits, labels, 1)
        top5_op = tf.nn.in_top_k(logits, labels, 5)

        # initialization
        self._session.run(tf.global_variables_initializer())

        # load checkpoint
        system = self.config.system
        checkpoint = CheckpointHandler(
            self._session, system.checkpoint.load, system.checkpoint.save,
            system.search_paths.checkpoints)
        checkpoint.load()

        # queue runners
        coord = tf.train.Coordinator()
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            queue_threads = qr.create_threads(
                self._session, coord=coord, daemon=True, start=True)
            threads += queue_threads
        num_examples = self.config.dataset.num_examples_per_epoch.validate
        batch_size = self.config.system.batch_size
        num_iterations = math.ceil(num_examples / batch_size)
        num_final_examples = num_examples % batch_size

        log.info('Starting evaluation...')
        top1s, top5s, step, total = 0.0, 0.0, 0, 0
        try:
            while step < num_iterations and not coord.should_stop():
                top1, top5 = self._session.run([top1_op, top5_op])
                if step == num_iterations - 1:
                    # final iteration
                    top1 = top1[:num_final_examples]
                    top5 = top5[:num_final_examples]
                    total += num_final_examples
                else:
                    total += batch_size
                top1s += np.sum(top1)
                top5s += np.sum(top5)
                top1_acc = top1s / total
                top5_acc = top5s / total
                step += 1
                self._update_progress(step, top1_acc, top5_acc, num_iterations)
        except KeyboardInterrupt as e:
            log.info('Evaluation aborted.')
            coord.request_stop(e)
        else:
            log.info('Evaluation complete.')
            log.info('\ttop1: {:.2f}%, top5: {:.2f}% [{} images]'.format(
                top1_acc * 100, top5_acc * 100, total))
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

    def eval(self):
        log.info('Initializing...')
        with self._graph.as_default():
            self._session = tf.Session(graph=self._graph)
            self._eval()
