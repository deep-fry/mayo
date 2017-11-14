import tensorflow as tf

from mayo.util import Percent
from mayo.override import util
from mayo.override.base import OverriderBase, Parameter


class GaterBase(OverriderBase):
    def _info(self, session):
        # FIXME it doesn't make sense to run `gate` once as its density
        # varies from run to run.
        gate = util.cast(session.run(self.gate), int)
        density = Percent(util.sum(gate) / util.count(gate))
        return self._info_tuple(
            gate=self.gate.name, density=density, count_=gate.size)

    @classmethod
    def finalize_info(cls, table):
        densities = table.get_column('density')
        count = table.get_column('count_')
        avg_density = sum(d * c for d, c in zip(densities, count)) / sum(count)
        footer = [None, '    overall: ', Percent(avg_density), None]
        table.set_footer(footer)


class RandomChannelGater(OverriderBase):
    def __init__(self, ratio=None, should_update=True):
        super().__init__(should_update)
        self.ratio = ratio

    def _apply(self, value):
        n, h, w, c = (int(d) for d in value.shape)
        # threshold
        omap = {'Sign': 'Identity'}
        # random gating
        random_number = tf.random_uniform(
            shape=[n, 1, 1, c], minval=self.ratio - 1, maxval=self.ratio)
        with tf.get_default_graph().gradient_override_map(omap):
            self.gate = tf.sign(random_number)
            self.gate = tf.clip_by_value(self.gate, 0, 1)
        # gates out feature maps with low vairance and replace the whole
        # feature map with its mean
        tf.add_to_collection('mayo.gates', self.gate)
        return self.gate * value


class ChannelGater(OverriderBase):
    threshold = Parameter('threshold', 1, [], tf.float32)

    def __init__(self, threshold=None, policy=None, should_update=True):
        super().__init__(should_update)
        self.threshold = threshold
        self.policy = policy

    def _apply(self, value):
        policy = self.policy
        value_pool = tf.nn.relu(value)
        n, h, w, c = (int(d) for d in value.shape)
        pool_params = {
            'padding': 'VALID',
            'ksize': [1, h, w, 1],
            'strides': [1, 1, 1, 1]
        }
        if policy == 'avg' or policy is None:
            pooled = tf.nn.avg_pool(value_pool, **pool_params)
        if policy == 'max':
            pooled = tf.nn.max_pool(tf.abs(value_pool), **pool_params)
        if policy == 'mix':
            maxed = tf.nn.max_pool(tf.abs(value_pool), **pool_params)
            avged = tf.nn.avg_pool(tf.abs(value_pool), **pool_params)
            pooled = maxed - avged
        #  mean, variance = tf.nn.moments(value, axes=[1, 2])
        #  variance = tf.reshape(variance, shape=[n, 1, 1, c])
        #  mean = tf.reshape(mean, shape=[n, 1, 1, c])
        # threshold
        # omap = {'Sign': 'Identity'}
        # with tf.get_default_graph().gradient_override_map(omap):
        #     self.gate = tf.sign(mean - self.threshold)
        #     self.gate = tf.clip_by_value(self.gate, 0, 1)
        # gates out feature maps with low vairance and replace the whole
        # feature map with its mean
        self.gate = util.cast(tf.abs(pooled) > self.threshold, float)
        self.pooled = pooled
        tf.add_to_collection('mayo.gates', self.gate)
        # return mean * (1 - self.gate) + self.gate * var
        return self.gate * value

    def _update(self, session):
        return
