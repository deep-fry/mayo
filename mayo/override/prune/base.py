import tensorflow as tf

from mayo.util import Percent
from mayo.override import util
from mayo.override.base import OverriderBase, Parameter


class PrunerBase(OverriderBase):
    mask = Parameter('mask', None, None, 'bool')

    def __init__(self, session, should_update=True):
        super().__init__(session, should_update)

    def _apply(self, value):
        self._parameter_config = {
            'mask': {
                'initial': tf.ones_initializer(dtype=tf.bool),
                'shape': value.shape,
            }
        }
        return value * util.cast(self.mask, float)

    def _updated_mask(self, var, mask):
        raise NotImplementedError(
            'Method to compute an updated mask is not implemented.')

    def _update(self):
        mask = self._updated_mask(self.before, self.mask)
        self.session.assign(self.mask, mask)

    def _info(self):
        mask = util.cast(self.session.run(self.mask), int)
        density = Percent(util.sum(mask) / util.count(mask))
        return self._info_tuple(
            mask=self.mask.name, density=density, count_=mask.size)

    @classmethod
    def finalize_info(cls, table):
        densities = table.get_column('density')
        count = table.get_column('count_')
        avg_density = sum(d * c for d, c in zip(densities, count)) / sum(count)
        footer = [None, '    overall: ', Percent(avg_density), None]
        table.add_row(footer)
        return footer


class ChannelPrunerBase(OverriderBase):
    mask = Parameter('mask', None, None, 'bool')

    def __init__(self, session, should_update=True):
        super().__init__(session, should_update)

    def _apply(self, value):
        # check shape
        if not len(value.shape) >= 3:
            raise ValueError(
                'Incorrect dimension {} for channel pruner'
                .format(value.shape))
        self.num_channels = value.shape[-1]
        self._parameter_config = {
            'mask': {
                'initial': tf.ones_initializer(dtype=tf.bool),
                'shape': self.num_channels,
            },
        }
        mask = tf.reshape(self.mask, (1, 1, 1, -1))
        return value * util.cast(mask, float)

    def _updated_mask(self, var, mask):
        raise NotImplementedError(
            'Method to compute an updated mask is not implemented.')

    def _update(self):
        mask = self._updated_mask(self.before, self.mask)
        self.session.assign(self.mask, mask)

    def _info(self):
        mask = util.cast(self.session.run(self.mask), int)
        density = Percent(util.sum(mask) / util.count(mask))
        return self._info_tuple(
            mask=self.mask.name, density=density, count_=mask.size)

    @classmethod
    def finalize_info(cls, table):
        densities = table.get_column('density')
        count = table.get_column('count_')
        avg_density = sum(d * c for d, c in zip(densities, count)) / sum(count)
        footer = [None, '    overall: ', Percent(avg_density), None]
        table.add_row(footer)
        return footer
