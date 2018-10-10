from mayo.log import log
from mayo.override import util
from mayo.override.base import Parameter
from mayo.override.quantize.base import QuantizerBase
import tensorflow as tf


class TernaryQuantizer(QuantizerBase):
    """
    Ternary quantization, quantizes all values into the range:
        {- 2^base * scale, 0, 2^base * scale}.

    Args:
        base: The universal coarse-grain scaling factor
              applied to tenary weights.
    References:
        - Extremely Low Bit Neural Network: Squeeze the Last Bit Out with ADMM
        - Trained Ternary Quantization
    """
    base = Parameter('base', 1, [], 'int', trainable=False)
    scale = Parameter('scale', 1.0, [], 'float', trainable=True)

    def __init__(
            self, session, base=None, stochastic=None,
            should_update=True, enable=True):
        super().__init__(session, should_update, enable)
        if base is not None:
            if base < 0:
                raise ValueError(
                    'Base of ternary quantization must be '
                    'greater or equal than 0.')
            self.base = base
        if stochastic is not None:
            raise NotImplementedError(
                'Ternary quantization does not implement stochastic mode.')

    def _quantize(self, value, base=None):
        scale = self.scale
        base = util.cast(self.base if base is None else base, int)
        shift = util.cast(2 ** base, float)
        positives = util.cast(value > 0, float)
        negatives = util.cast(value < 0, float)
        return positives * shift * scale - negatives * shift * scale

    def _apply(self, value):
        return self._quantize(value)

    def _info(self):
        base = int(self.eval(self.base))
        return self._info_tuple(width=2, base=base)


class ChannelTernaryQuantizer(TernaryQuantizer):
    """Same tenary quantization, but channel-wise scaling factors.  """
    scale = Parameter('scale', None, None, 'float', trainable=True)

    def _quantize(self, value, base=None):
        # @Aaron @Xitong FIXME possible redundancy:
        # this code is idenitical to super()._quantize()
        scale = self.scale
        base = util.cast(self.base if base is None else base, int)
        shift = util.cast(2 ** base, float)
        positives = util.cast(value > 0, float)
        negatives = util.cast(value < 0, float)
        # FIXME verify this multiplication is broadcasting correctly
        return positives * shift * scale - negatives * shift * scale

    def _apply(self, value):
        self._parameter_config = {
            'scale': {
                'initial': tf.ones_initializer(),
                # a vector that has a length matching
                # the number of output channels
                'shape': value.shape[-1],
            }
        }
        return self._quantize(value)
