import math
import collections

import scipy
import tensorflow as tf
from tensorflow.contrib import slim

from mayo.log import log
from mayo.util import memoize_method, Table, ensure_list, object_from_params
from mayo.net.base import NetBase
from mayo.net.util import ParameterTransformer, use_name_not_scope


class _NetBase(NetBase):
    """ Utility functions to create a TensorFlow network.  """

    def __init__(self, model, images, labels, num_classes, is_training, reuse):
        self.is_training = is_training
        self._transformer = ParameterTransformer(
            num_classes, is_training, reuse)
        super().__init__(model, {'input': images})
        self._labels = labels
        self._verify_io()

    def _verify_io(self):
        nodes = list(self._graph.input_nodes())
        if len(nodes) != 1 and nodes[0].name != 'input':
            raise ValueError(
                'We expect the graph to have a unique images input named '
                '"input", found {!r}.'.format(nodes))
        nodes = list(self._graph.output_nodes())
        if len(nodes) != 1 or nodes[0].name != 'output':
            raise ValueError(
                'We expect the graph to have a unique logits output named '
                '"output", found {!r}.'.format(nodes))

    def labels(self):
        return self._labels

    def logits(self):
        return self.outputs()['output']

    @property
    def overriders(self):
        return self._transformer.overriders

    def layers(self):
        return {n.name: self._tensors[n] for n in self._graph.layer_nodes()}

    @memoize_method
    def loss(self):
        logits = self.logits()
        with tf.name_scope('loss'):
            labels = slim.one_hot_encoding(self.labels(), logits.shape[1])
            loss = tf.losses.softmax_cross_entropy(
                logits=logits, onehot_labels=labels)
            loss = tf.reduce_mean(loss)
        return loss

    def top(self, count=1):
        name = 'top_{}'.format(count)
        try:
            return self._tensors[name]
        except KeyError:
            pass
        top = tf.nn.in_top_k(self.logits(), self.labels(), count)
        self._tensors[name] = top
        return top

    def accuracy(self, top_n=1):
        name = 'accuracy_{}'.format(top_n)
        try:
            return self._tensors[name]
        except KeyError:
            pass
        top = self.top(top_n)
        acc = tf.reduce_sum(tf.cast(top, tf.float32))
        acc /= top.shape.num_elements()
        self._tensors[name] = acc
        return acc

    def info(self):
        var_info = Table(['variable', 'shape'])
        var_info.add_rows((v, v.shape) for v in tf.trainable_variables())
        var_info.add_column(
            'count', lambda row: var_info[row, 'shape'].num_elements())
        var_info.set_footer(
            ['', '    total:', sum(var_info.get_column('count'))])
        layer_info = Table(['layer', 'shape'])
        for name, tensors in self.layers().items():
            tensors = ensure_list(tensors)
            for tensor in tensors:
                layer_info.add_row((name, tensor.shape))
        return {'variables': var_info, 'layers': layer_info}

    def _params_to_text(self, params):
        arguments = []
        for k, v in params.items():
            try:
                v = '{}()'.format(v.__qualname__)
            except (KeyError, AttributeError):
                pass
            arguments.append('{}={}'.format(k, v))
        return '    ' + '\n    '.join(arguments)

    def _instantiate_numeric_padding(self, tensors, params):
        pad = params.get('padding')
        if not isinstance(pad, int):
            return tensors
        # disable pad for next layer
        params['padding'] = 'VALID'
        # 4D tensor NxHxWxC, pad H and W
        paddings = [[0, 0], [pad, pad], [pad, pad], [0, 0]]
        if isinstance(tensors, collections.Sequence):
            return [tf.pad(t, paddings) for t in tensors]
        else:
            return tf.pad(tensors, paddings)

    def _instantiate_layer(self, name, tensors, params, module_path):
        # transform parameters
        params, scope = self._transformer.transform(name, params, module_path)
        with scope:
            layer_type = params['type']
            layer_key = tf.get_variable_scope().name
            log.debug(
                'Instantiating {!r} of type {!r} with arguments:\n{}'
                .format(layer_key, layer_type, self._params_to_text(params)))
            tensors = self._instantiate_numeric_padding(tensors, params)
            return super()._instantiate_layer(
                name, tensors, params, module_path)


class Net(_NetBase):
    """ Create a TensorFlow graph from "config.model" model definition.  """

    def instantiate_convolution(self, tensor, params):
        return slim.conv2d(tensor, **params)

    def instantiate_depthwise_separable_convolution(self, tensor, params):
        scope = params.pop('scope')
        num_outputs = params.pop('num_outputs', None)
        # depthwise layer
        stride = params.pop('stride')
        kernel = params.pop('kernel_size')
        depth_multiplier = params.pop('depth_multiplier', 1)
        depthwise_regularizer = params.pop('depthwise_regularizer')
        depthwise = slim.separable_conv2d(
            tensor, num_outputs=None, kernel_size=kernel, stride=stride,
            weights_regularizer=depthwise_regularizer,
            depth_multiplier=1, scope=scope, **params)
        if num_outputs is None:
            # if num_outputs is None, it is a depthwise by default
            return depthwise
        # pointwise layer
        pointwise_regularizer = params.pop('pointwise_regularizer')
        num_outputs = max(int(num_outputs * depth_multiplier), 8)
        pointwise = slim.conv2d(
            depthwise, num_outputs=num_outputs, kernel_size=[1, 1], stride=1,
            weights_regularizer=pointwise_regularizer, scope=scope, **params)
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

    def instantiate_average_pool(self, tensor, params):
        self._reduce_kernel_size_for_small_input(params, tensor)
        return slim.avg_pool2d(tensor, **params)

    def instantiate_max_pool(self, tensor, params):
        self._reduce_kernel_size_for_small_input(params, tensor)
        return slim.max_pool2d(tensor, **params)

    def instantiate_fully_connected(self, tensor, params):
        return slim.fully_connected(tensor, **params)

    def instantiate_softmax(self, tensor, params):
        return slim.softmax(tensor, **params)

    def instantiate_dropout(self, tensor, params):
        params['is_training'] = self.is_training
        return slim.dropout(tensor, **params)

    def instantiate_local_response_normalization(self, tensor, params):
        return tf.nn.local_response_normalization(
            tensor, **use_name_not_scope(params))

    def instantiate_squeeze(self, tensor, params):
        return tf.squeeze(tensor, **use_name_not_scope(params))

    def instantiate_flatten(self, tensor, params):
        return slim.flatten(tensor, **params)

    def instantiate_gated_convolution(self, tensor, params):
        num, height, width, channels = tensor.shape
        policy = params.pop('policy')
        # convolution
        output = self.instantiate_convolution(tensor, params)
        # gating network
        gate_scope = '{}/gate'.format(params['scope'])
        # pool
        pool_params = {
            'padding': 'VALID',
            'kernel_size': [height, width],
            'scope': gate_scope,
        }
        # max pool is hardware-friendlier
        gate = self.instantiate_max_pool(tensor, pool_params)
        # fc
        num_outputs = params['num_outputs']
        fc_params = {
            'kernel_size': 1,
            'num_outputs': num_outputs,
            'biases_initializer': tf.ones_initializer(),
            'weights_initializer':
                tf.truncated_normal_initializer(stddev=0.01),
            'activation_fn': None,
            'scope': gate_scope,
        }
        gate = self.instantiate_convolution(gate, fc_params)
        # policies
        if policy.type == 'softmax_cross_entropy':
            # predictor policy
            # output pool
            _, out_height, out_width, out_channels = output.shape
            pool_params = {
                'padding': 'VALID',
                'kernel_size': [out_height, out_width],
                'scope': gate_scope,
            }
            # TODO is it really sensible to use averge pool as the
            # threshold criteria?
            avg_output = self.instantiate_max_pool(output, pool_params)
            # not training the output as we train the predictor `gate`
            avg_output = tf.stop_gradient(avg_output)
            # loss
            tf.losses.softmax_cross_entropy(
                avg_output, gate, weights=0.00001,
                loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
            # thresholding
            k = math.ceil(int(out_channels) * policy.density)
            gate_top_k, _ = tf.nn.top_k(gate, k=k, sorted=True)
            gate_threshold = tf.reduce_min(gate_top_k, axis=[1, 2, 3])
            # gate_max.shape [batch_size]
            for _ in range(3):
                gate_threshold = tf.expand_dims(gate_threshold, -1)
            # gate_max.shape [batch_size, 1, 1, 1]
            gate = tf.cast(gate > gate_threshold, tf.float32)
            # training happens in softmax_cross_entropy
            gate = tf.stop_gradient(gate)
        else:
            # regularizer policy
            regu_cls, regu_params = object_from_params(policy)
            regularization = regu_cls(**regu_params)(gate)
            tf.add_to_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES, regularization)
            # threshold
            omap = {'Sign': 'Identity'}
            with tf.get_default_graph().gradient_override_map(omap):
                gate = tf.sign(gate)
                gate = tf.clip_by_value(gate, 0, 1)
        tf.add_to_collection('mayo.gates', gate)
        # gating
        return output * gate

    def instantiate_hadamard(self, tensor, params):
        # hadamard matrix
        channels = int(tensor.shape[3])
        if 2 ** int(math.log(channels, 2)) != channels:
            raise ValueError(
                'Number of channels must be a power of 2 for hadamard layer.')
        hadamard = scipy.linalg.hadamard(channels)
        hadamard = tf.constant(hadamard, dtype=tf.float32)
        hadamard = hadamard / float(channels)  # normalization
        # flatten input tensor
        flattened = tf.reshape(tensor, [-1, channels])
        if params.get('scales', True):
            init = tf.truncated_normal_initializer(mean=1, stddev=0.001)
            channel_scales = tf.get_variable(
                name='channel_scale', shape=[channels], initializer=init)
            flattened = flattened * channel_scales
        # transform with hadamard
        transformed = flattened @ hadamard
        reshaped = tf.reshape(transformed, shape=tensor.shape)
        # activation
        activation_function = params.get('activation_fn', tf.nn.relu)
        if activation_function is not None:
            reshaped = activation_function(reshaped)
        return reshaped

    def instantiate_concat(self, tensors, params):
        return tf.concat(tensors, **use_name_not_scope(params))
