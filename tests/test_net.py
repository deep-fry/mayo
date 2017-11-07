from common import TestCase

import types
import itertools

import networkx as nx
import tensorflow as tf
from tensorflow.contrib import slim

from mayo.net.graph import Graph, TensorNode, LayerNode, JoinNode
from mayo.net.legacy import _InstantiationParamTransformer
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
                    'layers': {'inner': {'type': 'pool', 'value': '^value'}},
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
        # assert ^value properly replaced
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
        self.transformer = _InstantiationParamTransformer(
            self.num_classes, self.is_training)

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

    def test_empty_norm_scope(self):
        null_scope = slim.arg_scope([])
        test_scope = self.transformer._norm_scope({})
        self._assertScopeEqual(test_scope, null_scope)

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
        test_scope = self.transformer._norm_scope(params)
        self._assertScopeEqual(test_scope, bn_scope)

    def test_overrider_scope(self):
        params = {
            'biases_overrider': FixedPointQuantizer(),
            'weights_overrider': FixedPointQuantizer(),
        }
        scope = self.transformer._overrider_scope(params)
        with scope:
            v = tf.get_variable('test', [1])
            b = tf.get_variable('biases', [1])
            w = tf.get_variable('weights', [1])
        self.assertIsInstance(v, tf.Variable)
        self.assertEqual(v.op.name, 'test')
        self.assertEqual(len(self.transformer.overriders), 2)
        self.assertEqual(b, self.transformer.overriders[0].after)
        self.assertEqual(w, self.transformer.overriders[1].after)


class TestTensorFlowNet(TestCase):
    pass
