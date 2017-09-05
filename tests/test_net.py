import types
import unittest

import tensorflow as tf
from tensorflow.contrib import slim

from mayo.net import _InstantiationParamTransformer
from mayo.override import Rounder


class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.num_classes = 10
        self.is_training = True
        self.transformer = _InstantiationParamTransformer(
            self.num_classes, self.is_training)

    def _assertObjectEqual(self, x, y):
        self.assertEqual(x.__class__, y.__class__)
        self.assertEqual(
            getattr(x, '__dict__', None), getattr(y, '__dict__', None))

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
        overrider = {'type': 'mayo.override.Rounder'}
        overrider_object = Rounder()
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
                self._assertObjectEqual(value, initializer_object)
            elif 'regularizer' in key:
                self._assertObjectEqual(value, regularizer_object)
            elif 'overrider' in key:
                self._assertObjectEqual(value, overrider_object)

    def test_config_layer(self):
        params = {
            'num_outputs': 'num_classes',
            'name': 'test',
            'padding': 'valid',
            'activation_overrider': Rounder(),
        }
        self.transformer._config_layer(params)
        self.assertEqual(params['num_outputs'], self.num_classes)
        self.assertEqual(params['scope'], 'test')
        self.assertEqual(params['padding'], 'VALID')
        self.assertIsInstance(params['activation_fn'], types.FunctionType)

    def _assertScopeEqual(self, x, y):
        with x as x_scope:
            x = x_scope
        with y as y_scope:
            y = y_scope
        self._assertObjectEqual(x, y)

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
            'biases_overrider': Rounder(),
            'weights_overrider': Rounder(),
        }
        scope = self.transformer._overrider_scope(params)
        with scope:
            v = tf.get_variable('test', [1])
            b = tf.get_variable('biases', [1])
            w = tf.get_variable('weights', [1])
        self.assertIsInstance(v, tf.Variable)
        self.assertEqual(v.op.name, 'test')
        self.assertEqual(len(self.transformer.overriders), 2)
        self.assertEqual(b, self.transformer.overriders[0]._after)
        self.assertEqual(w, self.transformer.overriders[1]._after)
