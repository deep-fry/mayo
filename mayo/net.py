import copy
import functools
import itertools
import collections
from contextlib import contextmanager

import tensorflow as tf
from tensorflow.contrib import slim

from mayo.log import log
from mayo.util import (
    import_from_string, object_from_params, multi_objects_from_params, Table,
    ensure_list)
from mayo.override import ChainOverrider


def _null_scope():
    return slim.arg_scope([])


def one_to_one(inst_method):
    @functools.wraps(inst_method)
    def wrapper(self, tensors, params):
        if len(tensors) != 1:
            raise ValueError(
                'We expect exactly one input for {!r}'.format(inst_method))
        return [inst_method(self, tensors[0], params)]
    return wrapper


class _InstantiationParamTransformer(object):
    def __init__(self, num_classes, is_training):
        super().__init__()
        self.num_classes = num_classes
        self.is_training = is_training
        self.overriders = []

    @classmethod
    def _recursive_replace(cls, value, replace):
        if isinstance(value, str):
            if value.startswith('^'):
                return replace[value[1:]]
            return value
        if isinstance(value, list):
            return [cls._recursive_replace(v, replace) for v in value]
        if isinstance(value, collections.Mapping):
            for k, v in value.items():
                value[k] = cls._recursive_replace(v, replace)
            return value
        return value

    def _repace_module_kwargs(self, params):
        if params['type'] != 'module':
            return
        kwargs = params.get('kwargs', {})
        replace = {
            key: params.get(key, default_value)
            for key, default_value in kwargs.items()}
        layers = copy.deepcopy(params['layers'])
        params['layers'] = self._recursive_replace(layers, replace)

    def _create_hyperobjects(self, params):
        def _create_object_for_key(params, key):
            p = params.get(key, None)
            if p is None:
                return
            if 'overrider' in key:
                overriders = [
                    cls(**p) for cls, p in multi_objects_from_params(p)]
                if len(overriders) == 1:
                    params[key] = overriders[0]
                else:
                    params[key] = ChainOverrider(overriders)
            else:
                cls, p = object_from_params(p)
                params[key] = cls(**p)

        var_names = ['weights', 'biases']
        obj_names = ['regularizer', 'initializer', 'overrider']
        param_names = [
            '{}_{}'.format(v, o)
            for v, o in itertools.product(var_names, obj_names)]
        param_names += [
            'pointwise_regularizer', 'depthwise_regularizer',
            'activation_overrider']
        for name in param_names:
            _create_object_for_key(params, name)

    def _config_layer(self, name, params):
        # activation
        fn = params.get('activation_fn', None)
        if fn is not None:
            fn = import_from_string(fn)
            params['activation_fn'] = fn
        activation_overrider = params.pop('activation_overrider', None)
        if activation_overrider:
            params['activation_fn'] = lambda x: (fn or tf.nn.relu)(
                activation_overrider.apply(tf.get_variable, x))
        if activation_overrider:
            self.overriders.append(activation_overrider)

        # num outputs
        if params.get('num_outputs', None) == 'num_classes':
            params['num_outputs'] = self.num_classes
        # set up parameters
        params['scope'] = name
        try:
            params['padding'] = params['padding'].upper()
        except (KeyError, AttributeError):
            pass

    def _norm_scope(self, params):
        # we do not have direct access to normalizer instantiation,
        # so arg_scope must be used
        norm_params = params.pop('normalizer_fn', None)
        if not norm_params:
            return _null_scope()
        obj, norm_params = object_from_params(norm_params)
        norm_params['is_training'] = self.is_training
        params['normalizer_fn'] = obj
        return slim.arg_scope([params['normalizer_fn']], **norm_params)

    def _overrider_scope(self, params):
        biases_overrider = params.pop('biases_overrider', None)
        weights_overrider = params.pop('weights_overrider', None)

        def custom_getter(getter, *args, **kwargs):
            v = getter(*args, **kwargs)
            name = v.op.name
            overrider = None
            if 'biases' in name:
                overrider = biases_overrider
            elif 'weights' in name:
                overrider = weights_overrider
            if overrider is None:
                return v
            log.debug('Overriding {!r} with {!r}'.format(v.op.name, overrider))
            ov = overrider.apply(getter, v)
            self.overriders.append(overrider)
            return ov

        # we do not have direct access to variable creation,
        # so scope must be used
        scope = tf.get_variable_scope()
        return tf.variable_scope(scope, custom_getter=custom_getter)

    def transform(self, name, params):
        params = dict(params)
        # replace module kwargs with values
        self._repace_module_kwargs(params)
        # weight and bias hyperparams
        self._create_hyperobjects(params)
        # layer configs
        self._config_layer(name, params)
        # normalization arg_scope
        norm_scope = self._norm_scope(params)
        # overrider arg_scope
        overrider_scope = self._overrider_scope(params)
        return params, norm_scope, overrider_scope


class BaseNet(object):
    def __init__(self, config, images, labels, is_training, reuse=None):
        super().__init__()
        self.config = config
        self.is_training = is_training
        self._reuse = reuse
        self._transformer = _InstantiationParamTransformer(
            self.config.num_classes(), self.is_training)
        self.tensors = collections.OrderedDict()
        self.tensors['images'] = images
        self.tensors['labels'] = labels
        self.layers = collections.OrderedDict()
        self.instantiate()

    @contextmanager
    def context(self):
        var_ctx = tf.variable_scope(self.config.model.name, reuse=self._reuse)
        cpu_ctx = slim.arg_scope([slim.model_variable], device='/cpu:0')
        with var_ctx, cpu_ctx as scope:
            yield scope

    @property
    def overriders(self):
        return self._transformer.overriders

    def _use_name_not_scope(self, params):
        params['name'] = params.pop('scope')
        return params

    def _instantiate_numeric_padding(self, tensors, params):
        pad = params.get('padding')
        if not isinstance(pad, int):
            return tensors
        # disable pad for next layer
        params['padding'] = 'VALID'
        # 4D tensor NxHxWxC, pad H and W
        paddings = [[0, 0], [pad, pad], [pad, pad], [0, 0]]
        return [tf.pad(t, paddings) for t in tensors]

    def _instantiate_edge(self, edge, layers, iodef, module):
        location = 'module {!r}'.format(module) if module else 'graph'
        try:
            from_tensors = ensure_list(edge['from'])
            to_tensors = ensure_list(edge['to'])
            with_layers = ensure_list(edge['with'])
        except KeyError:
            raise KeyError(
                'Graph edge definition expects keys "from", "with" and "to", '
                'the edge {} in {} lacks at least one of the above.'
                .format(edge, module))
        log.debug(
            'Instantiating edge from [{}] to [{}] with layers [{}] in {}...'
            .format(
                ', '.join(from_tensors), ', '.join(to_tensors),
                ', '.join(with_layers), location))
        tensors = []
        for t in from_tensors:
            try:
                tensors.append(iodef[t])
            except KeyError:
                raise KeyError(
                    'Tensor named {!r} is not defined in {}.'
                    .format(t, location))
        for layer_name in with_layers:
            try:
                params = layers[layer_name]
            except KeyError:
                raise KeyError(
                    'Layer definitions of {} does not contain a definition '
                    'for {!r}.'.format(location, layer_name))
            params, norm_scope, overrider_scope = \
                self._transformer.transform(layer_name, params)
            # get method by its name to instantiate a layer
            layer_type = params['type']
            func, params = object_from_params(params, self, 'instantiate_')
            # module scope
            var_scope = tf.variable_scope(module) if module else _null_scope()
            # instantiation
            arguments = []
            for k, v in params.items():
                try:
                    v = '{}()'.format(v.__qualname__)
                except (KeyError, AttributeError):
                    pass
                arguments.append('{}={}'.format(k, v))
            arguments = '\n    '.join(arguments)
            with norm_scope, overrider_scope, var_scope:
                layer_key = '{}/{}'.format(
                    tf.get_variable_scope().name, layer_name)
                log.debug(
                    'Instantiating {!r} of type {!r} with arguments:\n    {}'
                    .format(layer_key, layer_type, arguments))
                tensors = self._instantiate_numeric_padding(tensors, params)
                tensors = func(tensors, params)
            # add to layers
            if layer_key in self.layers:
                raise KeyError('Layer {!r} already exists.'.format(layer_key))
            self.layers[layer_key] = tensors
        # add to iodef
        if len(to_tensors) != len(tensors):
            raise ValueError(
                'We expect the number of final layer outputs to match the '
                'expected size in "to" in edge definition in {}.'
                .format(location))
        for to_name, tensor in zip(to_tensors, tensors):
            iodef[to_name] = tensor

    def _instantiate_graph(self, graph, layers, iodef, module=''):
        if not isinstance(graph, list):
            graph = [graph]
        if not graph:
            raise ValueError('No edges specified in "model.graph".')
        for edge in graph:
            self._instantiate_edge(edge, layers, iodef, module)

    def instantiate(self):
        # force all Variables to reside on the CPU
        model = self.config.model
        with self.context():
            self._instantiate_graph(model.graph, model.layers, self.tensors)
        if 'logits' not in self.tensors:
            raise ValueError('Logits layer not specified.')

    def instantiate_module(self, tensors, params):
        scope = params.pop('scope')
        input_names = params.get('inputs', ['input'])
        # set up inputs
        if len(tensors) != len(input_names):
            raise ValueError(
                'Received number of inputs does not match module {!r} '
                'defined size.'.format(scope))
        iodef = {name: value for name, value in zip(input_names, tensors)}
        # graph instantiation
        self._instantiate_graph(
            params['graph'], params['layers'], iodef, scope)
        # set up outputs
        outputs = []
        for name in params.get('outputs', ['output']):
            try:
                outputs.append(iodef[name])
            except KeyError:
                raise KeyError(
                    'Output named missing {!r} from the module {!r}.'
                    .format(name, scope))
        return outputs

    def generic_instantiate(self, tensors, params):
        raise NotImplementedError

    def labels(self):
        return self.tensors['labels']

    def logits(self):
        return self.tensors['logits']

    def loss(self):
        try:
            return self.tensors['loss']
        except KeyError:
            pass
        labels = self.labels()
        logits = self.logits()
        with tf.name_scope('loss'):
            labels = slim.one_hot_encoding(labels, logits.shape[1])
            loss = tf.losses.softmax_cross_entropy(
                logits=logits, onehot_labels=labels)
            loss = tf.reduce_mean(loss)
            tf.add_to_collection('losses', loss)
        self.tensors['loss'] = loss
        return loss

    def top(self, count=1):
        name = 'top_{}'.format(count)
        try:
            return self.tensors[name]
        except KeyError:
            pass
        logits = self.tensors['logits']
        labels = self.tensors['labels']
        top = tf.nn.in_top_k(logits, labels, count)
        self.tensors[name] = top
        return top

    def accuracy(self, top_n=1):
        name = 'accuracy_{}'.format(top_n)
        try:
            return self.tensors[name]
        except KeyError:
            pass
        top = self.top(top_n)
        acc = tf.reduce_sum(tf.cast(top, tf.float32))
        acc /= top.shape.num_elements()
        self.tensors[name] = acc
        return acc

    def info(self):
        var_info = Table(['variable', 'shape'])
        var_info.add_rows((v, v.shape) for v in tf.trainable_variables())
        var_info.add_column(
            'count', lambda row: var_info[row, 'shape'].num_elements())
        var_info.set_footer(
            ['', '    total:', sum(var_info.get_column('count'))])
        layer_info = Table(['layer', 'shape'])
        for name, tensors in self.layers.items():
            for tensor in tensors:
                layer_info.add_row((name, tensor.shape))
        return {'variables': var_info, 'layers': layer_info}


class Net(BaseNet):
    @one_to_one
    def instantiate_convolution(self, tensor, params):
        return slim.conv2d(tensor, **params)

    @one_to_one
    def instantiate_depthwise_separable_convolution(self, tensor, params):
        scope = params.pop('scope')
        num_outputs = params.pop('num_outputs', None)
        stride = params.pop('stride')
        kernel = params.pop('kernel_size')
        depth_multiplier = params.pop('depth_multiplier', 1)
        depthwise_regularizer = params.pop('depthwise_regularizer')
        if num_outputs is not None:
            pointwise_regularizer = params.pop('pointwise_regularizer')
        # depthwise layer
        depthwise = slim.separable_conv2d(
            tensor, num_outputs=None, kernel_size=kernel, stride=stride,
            weights_regularizer=depthwise_regularizer,
            depth_multiplier=1, scope='{}_depthwise'.format(scope), **params)
        if num_outputs is None:
            # if num_outputs is none, it is a depthwise by default
            return depthwise
        # pointwise layer
        num_outputs = max(int(num_outputs * depth_multiplier), 8)
        pointwise = slim.conv2d(
            depthwise, num_outputs=num_outputs, kernel_size=[1, 1], stride=1,
            weights_regularizer=pointwise_regularizer,
            scope='{}_pointwise'.format(scope), **params)
        return pointwise

    @staticmethod
    def _reduce_kernel_size_for_small_input(params, tensor):
        shape = tensor.get_shape().as_list()
        if shape[1] is None or shape[2] is None:
            return
        kernel = params['kernel_size']
        if isinstance(kernel, int):
            kernel = [kernel, kernel]
        stride = params.get('stride', 1)
        params['kernel_size'] = [
            min(shape[1], kernel[0]), min(shape[2], kernel[1])]
        # tensorflow complains when stride > kernel size
        params['stride'] = min(stride, kernel[0], kernel[1])

    @one_to_one
    def instantiate_average_pool(self, tensor, params):
        self._reduce_kernel_size_for_small_input(params, tensor)
        return slim.avg_pool2d(tensor, **params)

    @one_to_one
    def instantiate_max_pool(self, tensor, params):
        self._reduce_kernel_size_for_small_input(params, tensor)
        return slim.max_pool2d(tensor, **params)

    @one_to_one
    def instantiate_fully_connected(self, tensor, params):
        return slim.fully_connected(tensor, **params)

    @one_to_one
    def instantiate_softmax(self, tensor, params):
        return slim.softmax(tensor, **params)

    @one_to_one
    def instantiate_dropout(self, tensor, params):
        params['is_training'] = self.is_training
        return slim.dropout(tensor, **params)

    @one_to_one
    def instantiate_local_response_normalization(self, tensor, params):
        return tf.nn.local_response_normalization(
            tensor, **self._use_name_not_scope(params))

    @one_to_one
    def instantiate_squeeze(self, tensor, params):
        return tf.squeeze(tensor, **self._use_name_not_scope(params))

    @one_to_one
    def instantiate_flatten(self, tensor, params):
        return slim.flatten(tensor, **params)

    @one_to_one
    def instantiate_hadamard(self, tensor, params):
        # hadamard matrix is rescaled
        # A channel wise scaling variable can be used
        import scipy
        # generate a hadmard matrix
        input_channels = dim = int(tensor.shape[3])
        output_channels = params.pop('num_outputs')
        use_scales = params.pop('scales')
        if output_channels % input_channels != 0:
            raise ValueError('inputs and outputs must have a multiply factor'
                                'of 2')
        duplications = int(output_channels / input_channels)
        # spawn hadamard matrix from scipy
        hadamard_matrix = scipy.linalg.hadamard(dim)
        hadamard_matrix = tf.constant(hadamard_matrix, dtype=tf.float32)
        # large channel scales lead to divergence, hence rescale
        hadamard_matrix = hadamard_matrix / float(dim)
        tensor_reshaped = tf.reshape(tensor, [-1, dim])
        if use_scales:
            init = tf.truncated_normal_initializer(mean=1,
                                                   stddev=0.001)
        tensors_out = []
        for i in range(duplications):
            if use_scales:
                channel_scales = tf.get_variable(name='channel_scale' + str(i),
                    shape=[dim], initializer=init)
                tensor_reshaped = tensor_reshaped * channel_scales
            tensors_out.append(tf.reshape(tf.matmul(tensor_reshaped,
                hadamard_matrix), shape=tensor.shape))
        tensors_out = tf.concat(tensors_out, 3)
        tensors_out = tf.nn.relu6(tensors_out)
        return tensors_out

    @one_to_one
    def instantiate_channel_gating(self, tensor, params):
        # downsample the channels
        kernel_size = int(tensor.shape[2])
        # pool
        pooled = tf.nn.pool(tensor, pooling_type='AVG', padding='VALID',
                            window_shape=[kernel_size, kernel_size])
        net_input = tf.reshape(pooled,
            shape=[int(pooled.shape[0]), int(pooled.shape[3])])
        # building the network
        init = tf.truncated_normal_initializer(stddev=0.01)
        output_dim = params.pop('num_outputs')
        scope = params.pop('scope')
        net_out = slim.fully_connected(net_input, num_outputs=output_dim,
            weights_initializer=init, activation_fn=None,
            scope= '{}_fc'.format(scope))
        omap = {"Sign": "Identity"}
        with tf.get_default_graph().gradient_override_map(omap):
            gating = tf.sign(net_out)
        gating = tf.clip_by_value(gating, 0, 1)
        gating = net_out
        tf.add_to_collection('GATING_LOSS', tf.reduce_sum(gating))
        return gating

    def instantiate_gating_mult(self, tensors, params):
        gating = tensors[1]
        gating = tf.reshape(tensors[1],
            [int(gating.shape[0]), 1, 1, int(gating.shape[1])])
        return [tf.multiply(tensors[0], gating)]

    def instantiate_concat(self, tensors, params):
        return [tf.concat(tensors, **self._use_name_not_scope(params))]
