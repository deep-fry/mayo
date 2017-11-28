from common import TestCase

import tensorflow as tf

from mayo.override.base import OverriderBase, Parameter


class VariableMock(object):
    def __init__(self, name, shape, initializer, dtype, trainable):
        super().__init__()
        self.name = name
        self.shape = shape
        self.initializer = initializer
        self.dtype = dtype
        self.trainable = trainable

    @property
    def op(self):
        class OpMock(object):
            @property
            def name(mock):
                return self.name
        return OpMock

    def __repr__(self):
        return '<VariableMock {!r}>'.format(self.__dict__)


class TestParameter(TestCase):
    def setUp(self):
        self.parameter = Parameter('test', 1, (), tf.int32, True)

        class Overrider(OverriderBase):
            param = self.parameter

            def _apply(self, value):
                return self.param

        self.overrider = Overrider()
        var = VariableMock('input', 1, (), tf.int32, True)
        self.overrider.apply('scope', self._get_variable, var)

    def _get_variable(self, name, shape, initializer, dtype, trainable):
        return VariableMock(name, shape, initializer, dtype, trainable)

    def test_list_parameter(self):
        expect_parameters = {
            'test': self.parameter,
        }
        self.assertDictEqual(self.overrider.parameters, expect_parameters)

    def test_parameter_variable_instantiation(self):
        var = self.overrider.param
        expect_var = VariableMock(
            'scope/Overrider.test', (), var.initializer, tf.int32, True)
        self.assertObjectEqual(var, expect_var)
