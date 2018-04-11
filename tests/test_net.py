from common import TestCase

import types
import itertools

import networkx as nx
import tensorflow as tf
from tensorflow.contrib import slim

from mayo.config import Config
from mayo.net.graph import Graph, TensorNode, LayerNode, JoinNode
from mayo.net.base import NetBase
from mayo.net.tf import TFNet
from mayo.net.tf.transform import ParameterTransformer
from mayo.override import FixedPointQuantizer


class TestGraph(TestCase):
    def _assert_graph_equal(self, graph, expected_paths):
        nodes = set(graph.nodes())
        expected_nodes = {n for p in expected_paths for n in p}
        # nodes match
        self.assertSetEqual(nodes, expected_nodes)
        # path correct
        paths = []
        io_nodes = itertools.product(graph.input_nodes(), graph.output_nodes())
        for i, o in io_nodes:
            paths += list(nx.all_simple_paths(graph._graph, i, o))
        self.assertEqual(len(paths), len(expected_paths))
        self.assertSetEqual({tuple(p) for p in paths}, expected_paths)
        return paths

    def test_simple(self):
        model = {
            'name': 'test',
            'layers': {'conv': None, 'pool': None},
            'graph':
                {'from': 'input', 'with': ['conv', 'pool'], 'to': 'output'},
        }
        expected_paths = {(
            TensorNode('input', []),
            LayerNode('conv', None, ['test']),
            LayerNode('pool', None, ['test']),
            TensorNode('output', []),
        )}
        self._assert_graph_equal(Graph(model), expected_paths)

    def test_module(self):
        model = {
            'name': 'test',
            'layers': {
                'mod': {
                    'type': 'module',
                    'kwargs': {'value': 1},
                    'layers': {'inner': {'type': 'pool', 'value': '^(value)'}},
                    'graph':
                        {'from': 'input', 'with': 'inner', 'to': 'output'},
                }
            },
            'graph': {'from': 'input', 'with': 'mod', 'to': 'output'},
        }
        expected_paths = {(
            TensorNode('input', []),
            LayerNode('inner', None, ['test', 'mod']),
            TensorNode('output', []),
        )}
        paths = self._assert_graph_equal(Graph(model), expected_paths)
        # assert ^(value) properly replaced
        self.assertEqual(paths[0][1].params['value'], 1)

    def test_convergence(self):
        model = {
            'name': 'test',
            'layers': {'conv': None, 'pool': None, 'concat': None},
            'graph': [
                {'from': 'input', 'with': ['conv'], 'to': 'a'},
                {'from': 'input', 'with': ['pool'], 'to': 'b'},
                {'from': ['a', 'b'], 'with': ['concat'], 'to': 'output'},
            ],
        }
        expected_paths = {
            (
                TensorNode('input', []),
                TensorNode('input', ['test']),
                LayerNode('conv', None, ['test']),
                JoinNode(['a', 'b'], ['test']),
                LayerNode('concat', None, ['test']),
                TensorNode('output', []),
            ),
            (
                TensorNode('input', []),
                TensorNode('input', ['test']),
                LayerNode('pool', None, ['test']),
                JoinNode(['a', 'b'], ['test']),
                LayerNode('concat', None, ['test']),
                TensorNode('output', []),
            ),
        }
        self._assert_graph_equal(Graph(model), expected_paths)


class TestTransformer(TestCase):
    def setUp(self):
        self.num_classes = 10
        self.is_training = True
        self.reuse = False
        self.transformer = ParameterTransformer(
            self.num_classes, self.is_training, self.reuse)

    def test_create_hyperobjects(self):
        initializer = {
            'type': 'tensorflow.constant_initializer',
            'value': 0,
        }
        initializer_object = tf.constant_initializer(value=0)
        regularizer = {
            'type': 'tensorflow.contrib.layers.l2_regularizer',
            'scale': 0.00004,
        }
        regularizer_object = tf.contrib.layers.l2_regularizer(scale=0.00004)
        overrider = {'type': 'mayo.override.FixedPointQuantizer'}
        overrider_object = FixedPointQuantizer()
        params = {
            'weights_initializer': initializer,
            'weights_regularizer': regularizer,
            'weights_overrider': overrider,
            'biases_initializer': initializer,
            'biases_regularizer': regularizer,
            'biases_overrider': overrider,
            'activation_overrider': overrider,
        }
        self.transformer._create_hyperobjects(params)
        for key, value in params.items():
            if 'initializer' in key:
                self.assertObjectEqual(value, initializer_object)
            elif 'regularizer' in key:
                self.assertObjectEqual(value, regularizer_object)
            elif 'overrider' in key:
                self.assertObjectEqual(value, overrider_object)

    def test_config_layer(self):
        params = {
            'num_outputs': 'num_classes',
            'name': 'test',
            'padding': 'valid',
            'activation_overrider': FixedPointQuantizer(),
        }
        self.transformer._config_layer(params['name'], params)
        self.assertEqual(params['num_outputs'], self.num_classes)
        self.assertEqual(params['scope'], 'test')
        self.assertEqual(params['padding'], 'VALID')
        self.assertIsInstance(params['activation_fn'], types.FunctionType)

    def _assertScopeEqual(self, x, y):
        with x as x_scope:
            x = x_scope
        with y as y_scope:
            y = y_scope
        self.assertObjectEqual(x, y)

    def test_batch_norm_scope(self):
        kwargs = {
            'center': True,
            'scale': True,
            'decay': 0.9997,
            'epsilon': 0.001,
        }
        bn_scope = slim.arg_scope([slim.batch_norm], **kwargs)
        kwargs.update(type='tensorflow.contrib.slim.batch_norm')
        params = {'normalizer_fn': kwargs}
        scopes = []
        self.transformer._add_norm_scope(params, scopes)
        self._assertScopeEqual(scopes[0], bn_scope)

    def test_overrider_scope(self):
        params = {
            'biases_overrider': FixedPointQuantizer(),
            'weights_overrider': FixedPointQuantizer(),
        }
        scopes = []
        self.transformer._add_var_scope(params, ['module'], scopes)
        with scopes[0]:
            v = tf.get_variable('test', [1])
            b = tf.get_variable('biases', [1])
            w = tf.get_variable('weights', [1])
        self.assertIsInstance(v, tf.Variable)
        self.assertEqual(v.op.name, 'module/test')
        self.assertEqual(len(self.transformer.overriders), 2)
        self.assertEqual(b, self.transformer.overriders[0].after)
        self.assertEqual(w, self.transformer.overriders[1].after)


class TestNetBase(TestCase):
    class Base(NetBase):
        def instantiate_identity(self, node, tensor, params):
            return tensor

        def instantiate_concat(self, node, tensors, params):
            return tf.concat(tensors, axis=-1)

    def test_propagation(self):
        model = {
            'name': 'test',
            'layers': {'layer': {'type': 'identity'}},
            'graph': {'from': 'input', 'with': 'layer', 'to': 'output'},
        }
        images = 'A'
        net = self.Base(model, {'input': images})
        self.assertDictEqual(net.outputs(), {'output': images})

    def test_module(self):
        model = {
            'name': 'test',
            'layers': {
                'mod': {
                    'type': 'module',
                    'layers': {'inner': {'type': 'identity'}},
                    'graph':
                        {'from': 'input', 'with': 'inner', 'to': 'output'},
                }
            },
            'graph': {'from': 'input', 'with': 'mod', 'to': 'output'},
        }
        images = 'A'
        net = self.Base(model, {'input': images})
        self.assertDictEqual(net.outputs(), {'output': images})

    def test_convergence(self):
        model = {
            'name': 'test',
            'layers': {
                'a': {'type': 'identity'},
                'b': {'type': 'identity'},
                'concat': {'type': 'concat'},
            },
            'graph': [
                {'from': 'input', 'with': ['a'], 'to': 'a'},
                {'from': 'input', 'with': ['b'], 'to': 'b'},
                {'from': ['a', 'b'], 'with': ['concat'], 'to': 'output'},
            ],
        }
        inputs = tf.ones([2, 3], dtype=tf.float32)
        net = self.Base(model, {'input': inputs})
        output = net.outputs()['output']
        self.assertSequenceEqual(output.shape, [2, 6])


class TestTFNet(TestCase):
    class Net(TFNet):
        def instantiate_variable(self, node, tensor, params):
            with tf.variable_scope(params['scope']):
                return tf.get_variable('var', [], tf.float32)

    def _test_scope(self, reuse=False):
        model = {
            'name': 'test',
            'layers': {'layer': {'type': 'variable'}},
            'graph': {'from': 'input', 'with': 'layer', 'to': 'output'},
        }
        net = self.Net(model, None, None, 10, False, reuse)
        variable = net.outputs()['output']
        self.assertEqual(variable.name, 'test/layer/var:0')
        return variable

    def test_scope(self):
        with tf.Graph().as_default():
            return self._test_scope()

    def test_reuse(self):
        with tf.Graph().as_default():
            var1 = self._test_scope(False)
            var2 = self._test_scope(True)
            self.assertEqual(var1, var2)

    def test_lenet5(self):
        config = Config()
        config.yaml_update('models/lenet5.yaml')
        images = tf.ones([1, 28, 28, 1], dtype=tf.float32)
        net = TFNet(config.model, images, None, 10, False, False)
        logits = net.logits()
        self.assertSequenceEqual(logits.shape, [1, 10])
