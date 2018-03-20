import tensorflow as tf

from mayo.override import util
from mayo.override.base import Parameter
from mayo.override.prune.base import PrunerBase


class FilterPruner(PrunerBase):
    density = Parameter('density', 0.0, [], 'float')
    mask = Parameter('mask', None, None, 'bool')

    def __init__(self, session, density=None, should_update=True):
        super().__init__(session, should_update)
        self.density = density

    def _apply(self, value):
        self._parameter_config = {
            'mask': {
                'initial': tf.ones_initializer(dtype=tf.bool),
                'shape': tf.TensorShape([value.shape[-2], value.shape[-1]]),
            }
        }
        return value * util.cast(self.mask, float)

    def _l1_norm(self, value):
        # compute l1 norm for each filter
        axes = len(value.shape)
        assert axes == 4
        # mean, var = tf.nn.moments(util.abs(tensor), axes=[0, 1])
        # mean = np.mean(value, axis=(0, 1))
        # var = np.var(value, axis=(0, 1))
        # return mean + util.sqrt(var)
        return util.sum(util.abs(value), axis=(0, 1))

    def _threshold(self, value, density):
        value = value.flatten()
        index = int(value.size * density)
        return sorted(value)[index]

    def _updated_mask(self, tensor, mask):
        value, mask, density = self.session.run([tensor, mask, self.density])
        l1_norm = self._l1_norm(value)
        # mean, var = tf.nn.moments(util.abs(tensor), axes=[0, 1])
        return l1_norm > self._threshold(l1_norm, density)

    def _info(self):
        _, mask, density, count = super()._info()
        density = self.session.run(self.density)
        return self._info_tuple(
            mask=mask, density=density, count_=count)

    @classmethod
    def finalize_info(cls, table):
        footer = super().finalize_info(table)
        table.set_footer([None] + footer)
