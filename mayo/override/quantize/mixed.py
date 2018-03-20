import tensorflow as tf

from mayo.util import object_from_params
from mayo.override import util
from mayo.override.base import Parameter
from mayo.override.quantize.base import QuantizerBase


class MixedQuantizer(QuantizerBase):
    """
    Mixed Precision should be implemented as the following:
    mask1 * precision1 + mask2 * precision2 ...
    The masks are mutually exclusive
    Currently supporting:
        1. making a loss to the reg term
        2. quantizer_maps contains parallel quantizers that each can have
        a different quantizer
        3. channel wise granuarity based on output channels
    TODO:
    provide _update()
    """
    interval = Parameter('interval', 0.1, [], 'float')
    channel_mask = Parameter('channel_mask', None, None, 'int')

    def __init__(self, session, quantizers, index=0,
                 should_update=True, reg_factor=0.0, interval=0.1):
        super().__init__(session, should_update)
        self.quantizer_maps = {}
        for key, item in dict(quantizers).items():
            cls, params = object_from_params(item)
            quantizer = cls(session, **params)
            self.quantizer_maps[key] = quantizer
        self.reg_factor = reg_factor
        # the quantizer that makes a loss for training
        self.quantizers = quantizers
        self.picked_quantizer = list(quantizers.keys())[index]
        # keep record of an index for update
        self.index = index

    def _apply(self, value):
        # making an quantization loss to reg loss
        self._parameter_config = {
            'channel_mask': {
                'initial': tf.zeros_initializer(tf.bool),
                'shape': value.shape[-1],
            }
        }
        quantized_values = self._quantize(value)
        self._quantization_loss(value, quantized_values[self.picked_quantizer])
        # on mask indicates the quantized values
        return self._combine_masks(value, quantized_values)

    def _quantize(self, value, mean_quantizer=False):
        quantized_values = {}
        for key, quantizer in dict(self.quantizer_maps).items():
            scope = '{}/{}'.format(self._scope, self.__class__.__name__ + key)
            quantized_values[key] = quantizer.apply(
                self.node, scope, self._original_getter, value)
        return quantized_values

    def _combine_masks(self, value, quantized_values):
        """
        Args:
            quantized_value: the current mask is working on this current
                quantized value, this value is not included in
                quantizer_maps
        """
        if self.quantizer_maps:
            index = 0
            for key, quantizer in self.quantizer_maps.items():
                mask_label = index + 1
                channel_mask = util.cast(
                    util.equal(self.channel_mask, mask_label), float)
                if index == 0:
                    result = quantized_values[key] * channel_mask
                else:
                    result += quantized_values[key] * channel_mask
            # now handel off_mask
        channel_mask = tf.reshape(
            self.channel_mask,
            [1, 1, 1, self.channel_mask.shape[0]])
        off_mask = util.cast(util.equal(channel_mask, 0), float)
        return value * off_mask + result

    def _quantization_loss(self, value, quantized_value):
        loss = tf.reduce_sum(tf.abs(value - quantized_value))
        loss *= self.reg_factor
        loss_name = tf.GraphKeys.REGULARIZATION_LOSSES
        tf.add_to_collection(loss_name, loss)

    def _new_mask(self, mask, value, quantized_value, interval):
        loss = util.abs(value - quantized_value)
        # check the ones that are not quantized
        mask = mask.reshape((1, 1, 1, mask.shape[0]))
        unquantized_mask = util.logical_not(mask)
        # TODO: mask shape is incorrect
        loss_vec = util.mean(loss * unquantized_mask, (0, 1, 2))
        # sort
        num_active = util.ceil(len(loss_vec) * interval)
        threshold = sorted(loss_vec)[num_active]
        if interval >= 1.0:
            return util.cast(unquantized_mask, float)
        new_mask = (unquantized_mask * loss) > threshold
        return util.cast(util.logical_or(new_mask, mask), float)

    def _update(self):
        # update only the selected index
        quantizer = self.quantizer_maps[self.picked_quantizer]
        mask, value, quantized_value, interval = self.session.run(
            [self.channel_mask, self.before, quantizer.after, self.interval])
        mask = mask == (self.index + 1)
        new_mask = self._new_mask(mask, value, quantized_value, interval)
        self.index += 1
        self.picked_quantizer = list(self.quantizers.keys())[self.index]
