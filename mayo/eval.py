import time
import math

import numpy as np
import tensorflow as tf

from mayo.net import Net
from mayo.checkpoint import CheckpointHandler
from mayo.preprocess import Preprocess


class Evaluate(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._graph = tf.Graph()
        self._preprocessor = Preprocess(self.config)

    def logits(self, images, labels, reuse):
        self._net = Net(
            self.config, images, labels, True, graph=self._graph, reuse=reuse)
        return self._net.logits()

    def _update_progress(self, step, top1, top5, num_iterations):
        now = time.time()
        duration = now - getattr(self, '_prev_time', now)
        if duration != 0:
            num_steps = step - getattr(self, '_prev_step', step)
            imgs_per_sec = self.config.system.batch_size * num_steps
            imgs_per_sec /= float(duration)
            percentage = step / num_iterations * 100
            print('[{:.2f}%] top1: {:.2f}%, top5: {:.2f}% ({:.1f} imgs/sec)'
                  .format(percentage, top1 * 100, top5 * 100, imgs_per_sec))
        self._prev_time = now
        self._prev_step = step

    def _eval(self):
        images, labels = self._preprocessor.inputs(mode='validate')
        logits = self.logits(images, labels, False)
        top1_op = tf.nn.in_top_k(logits, labels, 1)
        top5_op = tf.nn.in_top_k(logits, labels, 5)

        # load checkpoint
        checkpoint = CheckpointHandler(
            self._session, self.config.name, self.config.dataset.name)
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
        num_iterations = int(math.ceil(num_examples / batch_size))

        print('Starting evaluation...')
        top1s, top5s, step = 0.0, 0.0, 0
        try:
            while step < num_iterations and not coord.should_stop():
                top1, top5 = self._session.run([top1_op, top5_op])
                top1s += np.sum(top1)
                top5s += np.sum(top5)
                step += 1
                total = step * batch_size
                top1_acc = top1s / total
                top5_acc = top5s / total
                if step % 20 == 0:
                    self._update_progress(
                        step, top1_acc, top5_acc, num_iterations)
        except KeyboardInterrupt as e:
            print('Evaluation aborted')
            coord.request_stop(e)

        print('Evaluation complete')
        print('\ttop1: {:.2f}%, top5: {:.2f}% [{} images]'
              .format(top1_acc * 100, top5_acc * 100, total))
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

    def eval(self):
        with self._graph.as_default():
            self._session = tf.Session(graph=self._graph)
            with self._session:
                self._eval()
