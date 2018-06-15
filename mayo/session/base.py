import functools
from contextlib import contextmanager

import tensorflow as tf

from mayo.log import log
from mayo.util import (
    memoize_property, flatten, object_from_params,
    Change, Table, Percent, unknown, format_shape)
from mayo.estimate import ResourceEstimator
from mayo.override import ChainOverrider
from mayo.net.base import LayerNode, JoinNode
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


class SessionBase(object, metaclass=SessionMeta):
    mode = None

    def __init__(self, config):
        super().__init__()
        log.debug('Instantiating...')
        # the default graph is made read-only to ensure
        # we always write to our graph
        default_graph = tf.get_default_graph()
        default_graph.finalize()
        self._ops = []
        self.extra_train_ops = {}
        self.finalizers = {}
        self.config = config
        self.change = Change()
        self.tf_graph = tf.Graph()
        self.initialized_variables = []
        self._assign_operators = {}
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        self.tf_session = tf.Session(graph=self.tf_graph, config=tf_config)
        self.tf_session.mayo_session = self
        self.checkpoint = CheckpointHandler(
            self.tf_session, config.system.search_path.checkpoint)
        self.estimator = ResourceEstimator(config.system.batch_size_per_gpu)
        self._register_progress()
        self._instantiate_task()
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

    @property
    def _task_constructor(self):
        return object_from_params(self.config.dataset.task)

    def _instantiate_task(self):
        task_cls, task_params = self._task_constructor
        self.task = task_cls(self, **task_params)

    def _finalize(self):
        # ensure configuration variable is instantiated
        self._config_var
        # invoke finalizers
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
        if self.mode == 'test':
            return len(self.task._preprocessor.files)
        return self.config.dataset.num_examples_per_epoch[self.mode]

    @property
    def num_gpus(self):
        return self.config.system.num_gpus

    def _tf_scalar(self, name, dtype=tf.int64):
        if dtype in [tf.int32, tf.int64, tf.float32, tf.float64, tf.bool]:
            initializer = tf.zeros_initializer()
        elif dtype is tf.string:
            initializer = tf.constant_initializer('')
        return tf.get_variable(
            name, [], initializer=initializer,
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

    def global_variables(self):
        return tf.global_variables()

    def trainable_variables(self):
        return tf.trainable_variables()

    @property
    def variables(self):
        return self.task.nets[0].variables

    @property
    def overriders(self):
        return self.task.nets[0].overriders

    def get_collection(self, key, first_gpu=False):
        func = lambda net, *args: tf.get_collection(key)
        collections = list(self.task.map(func))
        if first_gpu:
            return collections[0]
        return flatten(collections)

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

    @memoize_property
    def _config_var(self):
        return self._tf_scalar('mayo/config', dtype=tf.string)

    def save_checkpoint(self, name):
        # dump configuration to ensure we always know how
        # this checkpoint is trained
        self.assign(self._config_var, self.config.to_yaml())
        self.checkpoint.save(name)

    def info(self, plumbing=False):
        net = self.task.nets[0]
        info_dict = net.info(plumbing)
        # layer info
        stats = net.estimate()
        if plumbing:
            layer_info = {}
            for node, shape in net.shapes.items():
                stat = stats.get(node, {})
                stat['shape'] = list(shape)
                layer_info[node.formatted_name()] = stat
            info_dict['layers'] = layer_info
        else:
            keys = set()
            for node, stat in stats.items():
                if isinstance(stat, list):
                    for each in stat:
                        keys |= set(each)
                elif isinstance(stat, dict):
                    keys |= set(stat)
                else:
                    raise TypeError('Unrecognized type.')
            keys = sorted(k for k in keys if not k.startswith('_'))
            layer_info = Table(['layer', 'shape'] + keys)
            for node, shape in net.shapes.items():
                if isinstance(node, LayerNode):
                    values = stats.get(node, {})
                    values = tuple(values.get(k, unknown) for k in keys)
                else:
                    values = tuple([unknown] * len(keys))
                if isinstance(node, JoinNode):
                    shape = ', '.join(format_shape(s) for s in shape)
                else:
                    shape = format_shape(shape)
                layer_info.add_row((node.formatted_name(), shape) + values)
            try:
                macs = sum(layer_info.get_column('macs'))
            except ValueError:
                pass
            else:
                formatted_footer = [''] * len(keys)
                formatted_footer[keys.index('macs')] = macs
                layer_info.set_footer(['', ''] + formatted_footer)
            info_dict['layers'] = layer_info
        if self.overriders:
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
        overriders = []
        for each in self.overriders.values():
            overriders += list(each.values())
        if plumbing:
            for o in flatten(overriders):
                info = list(o.info())
                info_dict.setdefault(o.__class__, []).append(info)
        else:
            for o in flatten(overriders):
                info = o.info()
                table = info_dict.setdefault(o.__class__, Table(info._fields))
                table.add_row(info)
            for cls, table in info_dict.items():
                cls.finalize_info(table)
        return info_dict

    def _overrider_assign_parameters(self):
        # parameter assignments in overriders
        for _, overriders in self.overriders.items():
            for o in overriders.values():
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

    def debug(self, tensors):
        def wrapped(t):
            __import__('ipdb').set_trace()
            return t
        self.estimator.register(
            tensors, 'debug', history=1, transformer=wrapped)
        return tensors

    def interact(self):
        from IPython import embed
        embed()

    def plot(self):
        from mayo.plot import Plot
        Plot(self, self.config).plot()
