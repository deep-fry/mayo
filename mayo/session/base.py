import os
import re
import functools
import subprocess
from contextlib import contextmanager

import tensorflow as tf

from mayo.log import log
from mayo.util import memoize_property, flatten, Change, Table, Percent
from mayo.net.tf import TFNet
from mayo.estimate import ResourceEstimator
from mayo.override import ChainOverrider
from mayo.preprocess import Preprocess
from mayo.session.checkpoint import CheckpointHandler


class ReadOnlyGraphChangedError(Exception):
    """Graph should be read-only, but changes are made to the graph.  """


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
            method = cls.__dict__.get(name)
            if method is None:
                continue
            if isinstance(method, (staticmethod, classmethod)):
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
        self._ops = []
        self.extra_train_ops = {}
        self.finalizers = {}
        self.config = config
        self.change = Change()
        self._init_gpus()
        self.tf_graph = tf.Graph()
        self.initialized_variables = []
        self._assign_operators = {}
        self.tf_session = tf.Session(
            graph=self.tf_graph,
            config=tf.ConfigProto(allow_soft_placement=True))
        self.tf_session.mayo_session = self
        self.preprocessor = Preprocess(
            self.tf_session, self.mode, config, self.num_gpus)
        self.checkpoint = CheckpointHandler(
            self.tf_session, config.system.search_path.checkpoint)
        self.estimator = ResourceEstimator(config.system.batch_size_per_gpu)
        self._register_progress()
        self.nets = self._instantiate_nets()
        self._register_estimates()
        self._finalize()

    def __del__(self):
        log.debug('Finishing...')
        self.tf_session.close()

    def _register_progress(self):
        # progress
        def progress_formatter(estimator):
            progress = estimator.get_value('imgs_seen') / self.num_examples
            if self.is_training:
                return 'epoch: {:.2f}'.format(progress)
            if progress > 1:
                progress = 1
            return '{}: {}'.format(self.mode, Percent(progress))
        self.estimator.register(
            self.imgs_seen_op, 'imgs_seen',
            history=1, formatter=progress_formatter)

    def _register_estimates(self):
        # labels
        def label_transformer(index):
            # TODO label transformer
            return index
        history = 'infinite' if self.mode == 'validate' else None
        self.estimator.register(
            self.nets[0].labels(), 'labels', history=history,
            transformer=label_transformer)

    def _finalize(self):
        for name, finalizer in self.finalizers.items():
            log.debug(
                'Finalizing session with finalizer {!r}: {!r}'
                .format(name, finalizer))
            finalizer()

    @property
    def is_training(self):
        return self.mode == 'train'

    @property
    def batch_size(self):
        return self.config.system.batch_size_per_gpu * self.num_gpus

    @property
    def num_examples(self):
        return self.config.dataset.num_examples_per_epoch[self.mode]

    @property
    def num_gpus(self):
        return self.config.system.num_gpus

    def _auto_select_gpus(self):
        mem_bound = self.config.system.gpu_mem_bound
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

    def _tf_scalar(self, name, dtype=tf.int64):
        return tf.get_variable(
            name, [], initializer=tf.constant_initializer(0),
            trainable=False, dtype=dtype)

    @memoize_property
    def imgs_seen(self):
        return self._tf_scalar('imgs_seen', tf.int64)

    @memoize_property
    def imgs_seen_op(self):
        return tf.assign_add(self.imgs_seen, self.batch_size)

    @memoize_property
    def num_steps(self):
        return self.imgs_seen / self.batch_size

    @memoize_property
    def num_epochs(self):
        return self.imgs_seen / self.num_examples

    @memoize_property
    def num_examples_per_epoch(self):
        return self.config.dataset.num_examples_per_epoch[self.mode]

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

    def get_collection(self, key):
        func = lambda net: tf.get_collection(key)
        return flatten(self.net_map(func))

    def assign(self, var, tensor, raw_run=False):
        """
        Variable assignment.

        It uses placeholder for feeding values to assign, by doing so it avoids
        adding a `tf.assign` every time we make a new assignment.
        """
        try:
            op, placeholder = self._assign_operators[var]
        except KeyError:
            name = 'mayo/placeholder/{}'.format(var.op.name)
            placeholder = tf.placeholder(
                var.dtype, shape=var.get_shape(), name=name)
            op = tf.assign(var, placeholder)
            self._assign_operators[var] = op, placeholder
        run_func = self.raw_run if raw_run else self.run
        if isinstance(tensor, (tf.Variable, tf.Tensor)):
            tensor = run_func(tensor)
        run_func(op, feed_dict={placeholder: tensor})

    def load_checkpoint(self, name):
        # flush overrider parameter assignment
        self._overrider_assign_parameters()
        # restore variables
        restore_vars = self.checkpoint.load(name)
        for v in restore_vars:
            if v not in self.initialized_variables:
                self.initialized_variables.append(v)

    def save_checkpoint(self, name):
        self.checkpoint.save(name)

    def info(self, plumbing=False):
        net = self.nets[0]
        info_dict = net.info(plumbing)
        # layer info
        stats = self.estimator.get_estimates(net)
        if plumbing:
            layer_info = {}
            for node, stat in stats.items():
                stat['shape'] = list(net.shapes[node])
                layer_info[node.formatted_name()] = stat
            info_dict['layers'] = layer_info
        else:
            keys = list({k for v in stats.values() for k in v})
            layer_info = Table(['layer', 'shape'] + keys)
            for node, values in stats.items():
                values = tuple(values.get(k, 0) for k in keys)
                layer_info.add_row(
                    (node.formatted_name(), net.shapes[node]) + values)
            formatted_footers = [
                sum(layer_info.get_column(k)) for k in keys]
            layer_info.set_footer(['', '    total:'] + formatted_footers)
            info_dict['layers'] = layer_info
        if self.nets[0].overriders:
            info_dict['overriders'] = self._overrider_info(plumbing)
        return info_dict

    def _overrider_info(self, plumbing=False):
        def flatten(overriders):
            for o in overriders:
                if isinstance(o, ChainOverrider):
                    yield from flatten(o)
                else:
                    yield o
        info_dict = {}
        if plumbing:
            for o in flatten(self.nets[0].overriders):
                info = tuple(o.info())
                info_dict.setdefault(o.__class__, []).append(info)
        else:
            for o in flatten(self.nets[0].overriders):
                info = o.info()
                table = info_dict.setdefault(o.__class__, Table(info._fields))
                table.add_row(info)
            for cls, table in info_dict.items():
                cls.finalize_info(table)
        return info_dict

    def _overrider_assign_parameters(self):
        # parameter assignments in overriders
        for o in self.nets[0].overriders:
            o.assign_parameters()

    @contextmanager
    def ensure_graph_unchanged(self, func_name):
        ops = self.tf_graph.get_operations()
        yield
        new_ops = self.tf_graph.get_operations()
        diff_ops = []
        diff_assignments = []
        if len(ops) != len(new_ops):
            for o in new_ops:
                if o in ops:
                    continue
                if o.type in ('Placeholder', 'Assign', 'NoOp'):
                    diff_assignments.append(o)
                    continue
                diff_ops.append(o)
        if diff_assignments:
            log.debug(
                '{} creates new assignment operations {}.'
                .format(func_name, diff_assignments))
        if diff_ops:
            raise ReadOnlyGraphChangedError(
                '{} adds new operations {} to a read-only graph.'
                .format(func_name, diff_ops))

    def raw_run(self, ops, **kwargs):
        return self.tf_session.run(ops, **kwargs)

    def run(self, ops, batch=False, **kwargs):
        # ensure variables are initialized
        uninit_vars = []
        for var in self.global_variables():
            if var not in self.initialized_variables:
                uninit_vars.append(var)
        if uninit_vars:
            desc = '\n    '.join(v.op.name for v in uninit_vars)
            log.warn('Variables are not initialized:\n    {}'.format(desc))
            self.raw_run(tf.variables_initializer(uninit_vars))
            self.initialized_variables += uninit_vars

        # assign overrider hyperparameters
        self._overrider_assign_parameters()

        # session run
        if batch:
            results, statistics = self.raw_run(
                (ops, self.estimator.operations), **kwargs)
            # update statistics
            self.estimator.append(statistics)
            text = self.estimator.format(batch_size=self.batch_size)
            log.info(text, update=True)
            if log.is_enabled('debug'):
                self.estimator.debug()
        else:
            results = self.raw_run(ops, **kwargs)
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
                    self, self.config.model, images, labels, num_classes,
                    bool(nets))
            nets.append(net)
        return nets

    def net_map(self, func):
        for i, net in enumerate(self.nets):
            with self._gpu_context(i):
                yield func(net)

    def interact(self):
        from IPython import embed
        embed()

    def plot(self):
        from mayo.plot import Plot
        Plot(self, self.config).plot()
