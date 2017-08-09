import tensorflow as tf
from tensorflow.contrib import slim


class BaseNet(object):
    def __init__(self, config):
        super().__init__()
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.config = config
        self.end_points = {}
        with self.graph.as_default():
            input_layer = self.input()
            with tf.variable_scope(self.config['name'], values=[input_layer]):
                # force all Variables to reside on the CPU
                cpu_context = slim.arg_scope(
                    [slim.model_variable, slim.variable], device='/cpu:0')
                with cpu_context:
                    self._instantiate(input_layer)

    def input_variable(self):
        """
        This method defines a variable supplying input data.
        TODO: replace the implementation later to do without a placeholder.
        """
        shape = (self.config.dataset.batch_size, ) + self.config.input_shape
        return tf.placeholder(tf.float32, shape=shape, name='input')

    def label_variable(self):
        """
        This method defines a variable supplying labels.
        TODO: replace the implementation later to do without a placeholder.
        """
        shape = (self.config.dataset.batch_size, self.num_classes)
        return tf.placeholder(tf.int32, shape=shape, name='labels')

    def input(self):
        input_layer = self.input_variable()
        self.end_points['input'] = input_layer
        return input_layer

    def _instantiate(self, input_layer):
        net = input_layer
        for layer in self.config.net:
            # layer configs
            layer_name = layer['name']
            layer_type = layer['type']
            # set up parameters
            params = layer.setdefault('params', {})
            params['scope'] = layer_name
            for key, value in self.config.default.get(layer_type, {}).items():
                params.setdefault(key, value)
            if params.pop('batch_norm', False):
                params['normalizer_fn'] = slim.batch_norm
            # instantiation function
            # invokes method by its name to instantiate a layer
            func_name = 'instantiate_' + layer_type
            inst_func = getattr(self, func_name, self.generic_instantiate)
            net = inst_func(net, params)
            # end points
            self.end_points[layer_name] = net
            if layer_name == self.config.logits:
                self.end_points['logits'] = net

    def generic_instantiate(self, net, params):
        raise NotImplementedError(
            'Instantiation method for layer named "{}" is not implemented.'
            .format(params['scope']))

    def loss_op(self):
        try:
            return self.end_points['loss']
        except KeyError:
            pass
        labels = self.label_variable()
        self.end_points['labels'] = labels
        labels = slim.one_hot_encoding(labels, self.num_classes)
        loss = tf.losses.softmax_cross_entropy(
            logits=self.end_points['logits'], onehot_labels=labels)
        self.end_points['loss'] = loss
        return loss

    def save_graph(self):
        writer = tf.summary.FileWriter(self.config['name'], self.graph)
        writer.close()


class Net(BaseNet):
    def instantiate_convolution(self, net, params):
        return slim.conv2d(net, **params)

    def instantiate_depthwise_separable_convolution(self, net, params):
        return slim.separable_conv2d(net, **params)

    @staticmethod
    def _reduced_kernel_size_for_small_input(tensor, kernel):
        shape = tensor.get_shape().as_list()
        if shape[1] is None or shape[2] is None:
            return kernel
        return [min(shape[1], kernel[0]), min(shape[2], kernel[1])]

    def instantiate_average_pool(self, net, params):
        params['kernel_size'] = self._reduced_kernel_size_for_small_input(
            net, params['kernel_size'])
        return slim.avg_pool2d(net, **params)

    def instantiate_dropout(self, net, params):
        return slim.dropout(net, **params)

    def instantiate_squeeze(self, net, params):
        params['name'] = params.pop('scope')
        return tf.squeeze(net, **params)

    def instantiate_softmax(self, net, params):
        return slim.softmax(net, **params)
