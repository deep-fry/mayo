import time
import math

import tensorflow as tf

from mayo.log import log
from mayo.net import Net
from mayo.util import delta, moving_metrics, memoize, object_from_params
from mayo.preprocess import Preprocess
from mayo.checkpoint import CheckpointHandler


def _global_step(dtype=tf.int32):
    return tf.get_variable(
        'global_step', [], initializer=tf.constant_initializer(0),
        trainable=False, dtype=dtype)


class Train(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._graph = tf.Graph()
        self._nets = []
        self._preprocessor = Preprocess(self.config)
        self._init()

    @property
    @memoize
    def global_step(self):
        return _global_step()

    @property
    @memoize
    def imgs_seen(self):
        return tf.get_variable(
            'imgs_seen', shape=[],
            initializer=tf.constant_initializer(0),
            trainable=False, dtype=tf.int64)

    @property
    @memoize
    def learning_rate(self):
        params = self.config.train.learning_rate
        lr_class, params = object_from_params(params)
        if lr_class is tf.train.piecewise_constant:
            step_name = 'x'
        else:
            step_name = 'global_step'
        params[step_name] = self.global_step
        return lr_class(**params)

    @property
    @memoize
    def optimizer(self):
        params = self.config.train.optimizer
        optimizer_class, params = object_from_params(params)
        return optimizer_class(self.learning_rate, **params)

    def tower_loss(self, images, labels, reuse):
        net = Net(
            self.config, images, labels, True, graph=self._graph, reuse=reuse)
        self._nets.append(net)
        return net.loss(), net.accuracy()

    @staticmethod
    def _average_gradients(tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                # add 0 dimension to the gradients to represent the tower
                g = tf.expand_dims(g, 0)
                grads.append(g)
            # average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
            # simply return the first tower's pointer to the Variable
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def _setup_gradients(self):
        config = self.config.system
        # ensure batch size is divisible by number of gpus
        if config.batch_size % config.num_gpus != 0:
            raise ValueError(
                'Batch size must be divisible by number of devices')
        # initialize images and labels
        images_splits, labels_splits = self._preprocessor.preprocess_train()
        # for each gpu
        iterator = enumerate(zip(images_splits, labels_splits))
        tower_grads = []
        reuse = None
        for i, (images_split, label_split) in iterator:
            log.debug('Instantiating loss for GPU #{}.'.format(i))
            # loss with the proper nested contexts
            name = 'tower_{}'.format(i)
            with tf.device('/gpu:{}'.format(i)), tf.name_scope(name):
                # loss from the final tower
                self._loss, self._acc = self.tower_loss(
                    images_split, label_split, reuse)
                reuse = True
                # batch norm updates from the final tower
                with self._graph.as_default():
                    # summaries from the final tower
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
                    self._batch_norm_updates = tf.get_collection(
                        tf.GraphKeys.UPDATE_OPS, name)
                # gradients from all towers
                grads = self.optimizer.compute_gradients(self._loss)
                tower_grads.append(grads)
        self._gradients = self._average_gradients(tower_grads)
        # update num imgs
        self._imgs_seen_op = tf.assign_add(self.imgs_seen, config.batch_size)
        # summaries
        summaries += [
            tf.summary.scalar('learning_rate', self.learning_rate),
            tf.summary.scalar('loss', self._loss)]
        self._summary_op = tf.summary.merge(summaries)

    def _setup_train_operation(self):
        app_grad_op = self.optimizer.apply_gradients(
            self._gradients, global_step=self.global_step)
        ops = [app_grad_op]
        decay = self.config.train.get('moving_average_decay', 0)
        if decay:
            # instantiate moving average if moving_average_decay is supplied
            var_avgs = tf.train.ExponentialMovingAverage(
                self.config.train.moving_average_decay, self.global_step)
            var_avgs_op = var_avgs.apply(
                tf.trainable_variables() + tf.moving_average_variables())
            ops.append(var_avgs_op)
        bn_op = tf.group(*self._batch_norm_updates)
        ops.append(bn_op)
        self._train_op = tf.group(*ops)

    def _init_session(self):
        # build an initialization operation to run below
        init = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        self._session = tf.Session(config=config)
        self._session.run(init)

    def _init(self):
        log.info('Instantiating...')
        with self._graph.as_default():
            self._setup_gradients()
            self._setup_train_operation()
            log.info('Initializing session...')
            self._init_session()
            # checkpoint
            system = self.config.system
            self._checkpoint = CheckpointHandler(
                self._session, system.checkpoint.load, system.checkpoint.save,
                system.search_paths.checkpoints)
            self._checkpoint.load()
            tf.train.start_queue_runners(sess=self._session)
            self._nets[0].save_graph()

    def _update_progress(self, epoch, loss, accuracy, cp_epoch):
        metric_count = self.config.system.metrics_history_count
        if not isinstance(cp_epoch, str):
            cp_epoch = '{:.2f}'.format(cp_epoch)
        info = 'epoch: {:.2f} | loss: {:10f}{:5}'
        info += ' | acc: {:5.2f}% | ckpt: {}'
        loss_mean, loss_std = moving_metrics(
            'train.loss', loss, over=metric_count)
        loss_std = '±{}%'.format(int(loss_std / loss_mean * 100))
        acc_percent = sum(accuracy) / self.config.system.batch_size
        acc_percent *= self.config.system.num_gpus * 100
        acc_mean = moving_metrics(
            'train.accuracy', acc_percent, std=False, over=metric_count)
        info = info.format(epoch, loss_mean, loss_std, acc_mean, cp_epoch)
        # performance
        interval = delta('train.duration', time.time())
        if interval != 0:
            imgs = epoch * self.config.dataset.num_examples_per_epoch.train
            imgs_per_step = delta('train.step.imgs', imgs)
            imgs_per_sec = imgs_per_step / float(interval)
            imgs_per_sec = moving_metrics(
                'train.imgs_per_sec', imgs_per_sec,
                std=False, over=metric_count)
            info += ' | tp: {:4.0f}/s'.format(imgs_per_sec)
        log.info(info, update=True)

    @property
    @memoize
    def _summary_writer(self):
        path = self.config.system.search_paths.summaries[0]
        return tf.summary.FileWriter(path, graph=self._graph)

    def _save_summary(self, epoch):
        summary = self._session.run(self._summary_op)
        self._summary_writer.add_summary(summary, epoch)

    def once(self):
        tasks = [
            self._train_op, self._loss, self._acc, self._imgs_seen_op]
        _, loss, acc, imgs_seen = self._session.run(tasks)
        return loss, acc, imgs_seen

    def update_overriders(self):
        with self._graph.as_default():
            ops = []
            for n in self._nets:
                ops += n.update_overriders()
            log.info('Updating overrider variables...')
            self._session.run(ops)

    def train(self):
        imgs_per_epoch = self.config.dataset.num_examples_per_epoch.train
        # init
        log.info('Training start.')
        epoch = self._session.run(self.imgs_seen) / imgs_per_epoch
        cp_epoch = math.floor(epoch)
        # train iterations
        system = self.config.system
        max_epochs = system.max_epochs
        try:
            while epoch <= max_epochs:
                loss, acc, imgs_seen = self.once()
                epoch = imgs_seen / imgs_per_epoch
                if math.isnan(loss):
                    raise ValueError('Model diverged with a nan-valued loss.')
                self._update_progress(epoch, loss, acc, cp_epoch)
                summary_delta = delta('train.summary.epoch', epoch)
                if system.save_summary and summary_delta >= 0.1:
                    self._save_summary(epoch)
                cp_epoch = math.floor(epoch)
                if delta('train.checkpoint.epoch', cp_epoch) >= 1:
                    self._update_progress(epoch, loss, acc, 'saving')
                    with log.use_level('warn'):
                        self._checkpoint.save(cp_epoch)
        except KeyboardInterrupt:
            pass
        # interrupt
        try:
            log.info('Stopped.')
            timeout_secs = 3
            for i in range(timeout_secs):
                log.info(
                    'Saving checkpoint in {} seconds...'
                    .format(timeout_secs - i), update=True)
                time.sleep(1)
        except KeyboardInterrupt:
            log.info('We give up.')
            return
        self._checkpoint.save('latest')
