import tensorflow as tf
import numpy as np

from mayo.override.base import OverriderBase, Parameter


class LowRankApproximation(OverriderBase):
    singular = Parameter('singular', None, None, 'float')
    left = Parameter('left', None, None, 'float')
    right = Parameter('right', None, None, 'float')

    def __init__(self, session, should_update=True, ranks=0):
        super().__init__(session, should_update)
        # ranks to prune away
        self.ranks = ranks

    def _parameter_initial(self, value):
        dimensions = value.shape
        left_dimension = dimensions[0] * dimensions[2]
        right_dimension = dimensions[1] * dimensions[3]
        left_shape = (left_dimension, left_dimension)
        right_shape = (right_dimension, right_dimension)
        rows = int(left_dimension)
        columns = int(right_dimension)

        singular_shape = left_dimension if rows < columns else right_dimension

        self._parameter_config = {
            'singular': {
                'initial': tf.ones_initializer(dtype=tf.float32),
                'shape': singular_shape,
            },
            'left': {
                'initial': tf.ones_initializer(dtype=tf.float32),
                'shape': left_shape,
            },
            'right': {
                'initial': tf.ones_initializer(dtype=tf.float32),
                'shape': right_shape,
            }
        }
        return (rows, columns)

    def _apply(self, value):
        rows, columns = self._parameter_initial(value)
        if rows < columns:
            singular = tf.expand_dims(self.singular, 1) * tf.eye(rows, columns)
        else:
            singular = tf.expand_dims(self.singular, 0) * tf.eye(rows, columns)

        svd_construct = tf.matmul(
            tf.matmul(self.left, singular), self.right)
        return tf.reshape(svd_construct, value.shape)

    def _update(self):
        value = self.session.run(self.before)
        dimensions = value.shape
        if len(dimensions) == 4:
            meshed = np.reshape(
                value,
                [dimensions[0] * dimensions[2], dimensions[1] * dimensions[3]])
        elif len(dimensions) == 2:
            meshed = value
        else:
            raise ValueError('uh')
        left, singular, right = np.linalg.svd(meshed, full_matrices=True)
        singular[-self.ranks:] = 0.0
        self.session.assign(self.left, left)
        self.session.assign(self.singular, singular)
        self.session.assign(self.right, right)
