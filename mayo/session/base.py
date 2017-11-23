import os
import re
import time
import functools
import subprocess
import collections
from contextlib import contextmanager

import tensorflow as tf

from mayo.log import log
from mayo.util import (
    memoize_method, memoize_property, Change, flatten, Table)
from mayo.net.tf import TFNet
from mayo.override import ChainOverrider
from mayo.preprocess import Preprocess
from mayo.session.checkpoint import CheckpointHandler


class SessionMeta(type):
    """
    Automatically use the correct tf_session when invoking methods in Session.
    """
    def __new__(mcl, name, bases, nmspc):
        cls = super().__new__(mcl, name, bases, nmspc)
        for name in dir(cls):
            func = getattr(cls, name)
            if not callable(func):
                continue
            if name.startswith("__"):
                continue
            if isinstance(cls.__dict__.get(name), (staticmethod, classmethod)):
                continue
            setattr(cls, name, mcl.wrap(cls, func))
        return cls

    @staticmethod
    def wrap(cls, func):
        @functools.wraps(func)
        def wrapped(self, *args, **kwargs):
            try:
                session = self.tf_session
            except AttributeError:
                return func(self, *args, **kwargs)
            with session.as_default():
                with session.graph.as_default():
                    return func(self, *args, **kwargs)

        if getattr(func, '_wrapped', False):
            return func
        wrapped._wrapped = True
        return wrapped


class Session(object, metaclass=SessionMeta):
    mode = None

    def __init__(self, config):
        super().__init__()
        # the default graph is made read-only to ensure
        # we always write to our graph
        default_graph = tf.get_default_graph()
        default_graph.finalize()
        self.config = config
        self.change = Change()
        self._init_gpus()
        self.graph = tf.Graph()
        self.initialized_variables = []
        self.tf_session = tf.Session(
            graph=self.graph,
            config=tf.ConfigProto(allow_soft_placement=True))
        self.tf_session.mayo_session = self
        self.preprocessor = Preprocess(
            self.tf_session, self.mode, config, self.num_gpus)
        self.checkpoint = CheckpointHandler(
            self.tf_session, config.system.search_path.checkpoint)
        self.nets = self._instantiate_nets()
        self._to_update_op = collections.OrderedDict()
        self._to_update_formatter = {}

    def __del__(self):
        log.debug('Finishing...')
        self.tf_session.close()

    @property
    def batch_size(self):
        return self.config.system.batch_size_per_gpu * self.num_gpus

    @property
    def num_gpus(self):
        return self.config.system.num_gpus

    def _auto_select_gpus(self):
        mem_bound = 500
        try:
            info = subprocess.check_output(
                'nvidia-smi', shell=True, stderr=subprocess.STDOUT)
            info = re.findall('(\d+)MiB\s/', info.decode('utf-8'))
            log.debug('GPU memory usages (MB): {}'.format(', '.join(info)))
            info = [int(m) for m in info]
            gpus = [i for i in range(len(info)) if info[i] <= mem_bound]
        except subprocess.CalledProcessError:
            gpus = []
        if len(gpus) < self.num_gpus:
            log.warn(
                'Number of GPUs available {} is less than the number of '
                'GPUs requested {}.'.format(len(gpus), self.num_gpus))
        return ','.join(str(g) for g in gpus[:self.num_gpus])

    def _init_gpus(self):
        cuda_key = 'CUDA_VISIBLE_DEVICES'
        if cuda_key in os.environ:
            log.info('Using {}: {}'.format(cuda_key, os.environ[cuda_key]))
        gpus = self.config.system.get('visible_gpus', 'auto')
        if gpus != 'auto':
            if isinstance(gpus, list):
                gpus = ','.join(str(g) for g in gpus)
            else:
                gpus = str(gpus)
        else:
            gpus = self._auto_select_gpus()
        if gpus:
            log.info('Using GPUs: {}'.format(gpus))
        else:
            log.info('No GPUs available, using only one clone on CPU.')
            # FIXME hacky way to make it instantiate only one tower
            self.config.system.num_gpus = 1
        os.environ[cuda_key] = gpus

    def _tf_int(self, name, dtype=tf.int64):
        return tf.get_variable(
            name, [], initializer=tf.constant_initializer(0),
            trainable=False, dtype=dtype)

    @memoize_property
    def imgs_seen(self):
        return self._tf_int('imgs_seen', tf.int64)

    @memoize_property
    def num_steps(self):
        return self.imgs_seen / self.batch_size

    @memoize_property
    def num_epochs(self):
        imgs_per_epoch = self.config.dataset.num_examples_per_epoch.train
        return self.imgs_seen / imgs_per_epoch

    def _mean_metric(self, func):
        return tf.reduce_mean(list(self.net_map(func)))

    @memoize_property
    def accuracy(self):
        return self._mean_metric(lambda net: net.accuracy())

    @memoize_property
    def loss(self):
        # average loss without regularization, only for human consumption
        return self._mean_metric(lambda net: net.loss())

    def global_variables(self):
        return tf.global_variables()

    def trainable_variables(self):
        return tf.trainable_variables()

    def moving_average_variables(self):
        return tf.moving_average_variables()

    def get_collection(self, key):
        func = lambda net: tf.get_collection(key)
        return flatten(self.net_map(func))

    @memoize_method
    def moving_average_op(self):
        decay = self.config.get('train.moving_average_decay', 0)
        if not decay:
            return None
        # instantiate moving average if moving_average_decay is supplied
        var_avgs = tf.train.ExponentialMovingAverage(
            self.config.train.moving_average_decay, self.num_steps)
        avg_vars = tf.trainable_variables() + tf.moving_average_variables()
        return var_avgs.apply(avg_vars)

    def load_checkpoint(self, name):
        # flush overrider parameter assignment
        self.run([])
        # restore variables
        restore_vars = self.checkpoint.load(name)
        for v in restore_vars:
            if v not in self.initialized_variables:
                self.initialized_variables.append(v)

    def save_checkpoint(self, name):
        self.checkpoint.save(name)

    def info(self):
        info_dict = self.nets[0].info()
        if self.nets[0].overriders:
            info_dict['overriders'] = self.overrider_info()
        return info_dict

    def overrider_info(self):
        def flatten(overriders):
            for o in overriders:
                if isinstance(o, ChainOverrider):
                    yield from flatten(o)
                else:
                    yield o
        overrider_info = {}
        for o in flatten(self.nets[0].overriders):
            info = o.info(self)
            table = overrider_info.setdefault(o.__class__, Table(info._fields))
            table.add_row(info)
        for cls, table in overrider_info.items():
            cls.finalize_info(table)
        return overrider_info

    def _overrider_assign_parameters(self):
        # parameter assignments in overriders
        for o in self.nets[0].overriders:
            o.assign_parameters(self)

    def register_update(self, message, tensor, formatter=None):
        """
        Register message and tensor to print in progress update
        at each self.run().
        """
        self._to_update_op[message] = tensor
        self._to_update_formatter[message] = formatter

    @memoize_property
    def num_examples_per_epoch(self):
        return self.config.dataset.num_examples_per_epoch[self.mode]

    def _update_progress(self, to_update):
        if not to_update:
            return
        info = []
        for key, value in to_update.items():
            formatter = self._to_update_formatter[key]
            if formatter:
                value = formatter(value)
            info.append('{}: {}'.format(key, value))
        # performance
        epoch = to_update.get('epoch')
        interval = self.change.delta('step.duration', time.time())
        if interval != 0:
            if epoch:
                imgs = epoch * self.num_examples_per_epoch
                imgs_per_step = self.change.delta('step.images', imgs)
            else:
                imgs_per_step = self.batch_size
            imgs_per_sec = imgs_per_step / float(interval)
            imgs_per_sec = self.change.moving_metrics(
                'step.imgs_per_sec', imgs_per_sec, std=False)
            info.append('tp: {:4.0f}/s'.format(imgs_per_sec))
        log.info(' | '.join(info), update=True)

    def raw_run(self, ops, **kwargs):
        return self.tf_session.run(ops, **kwargs)

    @memoize_method
    def _register_update(self):
        for collection, func in self.nets[0].update_functions.items():
            func(self, collection)

    def run(self, ops, update_progress=False, **kwargs):
        # ensure variables are initialized
        uninit_vars = []
        for var in self.global_variables():
            if var not in self.initialized_variables:
                uninit_vars.append(var)
        if uninit_vars:
            desc = ', '.join(v.op.name for v in uninit_vars)
            log.warn('Variables are not initialized: {}'.format(desc))
            self.raw_run(tf.variables_initializer(uninit_vars))
            self.initialized_variables += uninit_vars

        # assign overrider hyperparameters
        self._overrider_assign_parameters()

        # extra statistics to print in progress update
        self._register_update()

        # session run
        filtered_to_update_op = {
            k: v for k, v in self._to_update_op.items()
            if isinstance(v, (tf.Tensor, tf.Variable))}
        results, to_update = self.raw_run(
            (ops, filtered_to_update_op), **kwargs)

        # progress update
        if update_progress:
            to_update = dict(self._to_update_op, **to_update)
            self._update_progress(to_update)

        return results

    def _preprocess(self):
        return self.preprocessor.preprocess()

    @contextmanager
    def _gpu_context(self, gid):
        with tf.device('/gpu:{}'.format(gid)):
            with tf.name_scope('tower_{}'.format(gid)) as scope:
                yield scope

    def _instantiate_nets(self):
        log.debug('Instantiating...')
        num_classes = self.config.num_classes()
        nets = []
        for i, (images, labels) in enumerate(self._preprocess()):
            log.debug('Instantiating graph for GPU #{}...'.format(i))
            with self._gpu_context(i):
                net = TFNet(
                    self.config.model, images, labels, num_classes,
                    self.mode == 'train', bool(nets))
            nets.append(net)
        return nets

    def net_map(self, func):
        for i, net in enumerate(self.nets):
            with self._gpu_context(i):
                yield func(net)

    def interact(self):
        from IPython import embed
        embed()
