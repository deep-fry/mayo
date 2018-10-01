from mayo.log import log
from mayo.override import util
from mayo.override.base import Parameter
from mayo.override.quantize.base import QuantizerBase


class TernaryQuantizer(QuantizerBase):
    """
    Quantize inputs into 2's compliment n-bit fixed-point values with d-bit
    dynamic range.

    Args:
        - base:
            The base shift number.
            Consider we have symmetric ternary weights with a scaling of alpah.
            base = 0:
                {+- 2^(0) * alpha, 0}
                {+- alpha, 0}
            base = 1:
                {+- 2^(1) * alpha, 0}
                {+- 2*alpha, 0}
    References:
        [1] Extremely Low Bit Neural Network: Squeeze the Last Bit Out with ADMM
        [2] TRAINED TERNARY QUANTIZATION
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
                    'Base of ternary quantization must be greater or equal than 0.')
            self.base = base 
        if stochastic is not None:
            raise ValueError(
                'Ternary quantization has no stochastic mode.')

    def _quantize(
            self, value, base=None):
        base = util.cast(self.base if base is None else base, int)
        scale = util.cast(self.scale, float)

        pos_values = util.cast((value > 0), float)
        neg_values = util.cast((value < 0), float)
        shift = util.cast(2 ** base, float)
        value = pos_values * shift * scale - neg_values * shift * scale
        return value

    def _apply(self, value):
        return self._quantize(value)

    def _info(self):
        base = int(self.eval(self.base))
        return self._info_tuple(width=2, base=base)


class ChannelTernaryQuantizer(TernaryQuantizer):
    """
    Quantize inputs into 2's compliment n-bit fixed-point values with d-bit
    dynamic range.

    Args:
        - base:
            The base shift number.
            Consider we have symmetric ternary weights with a scaling of alpah.
            base = 0:
                {+- 2^(0) * alpha, 0}
                {+- alpha, 0}
            base = 1:
                {+- 2^(1) * alpha, 0}
                {+- 2*alpha, 0}
    References:
        [1] Extremely Low Bit Neural Network: Squeeze the Last Bit Out with ADMM
        [2] TRAINED TERNARY QUANTIZATION
    """
    scale = Parameter('scale', None, None, 'float', trainable=True)

    def _quantize(
            self, value, base=None):
        base = util.cast(self.base if base is None else base, int)
        scale = util.cast(self.scale, float)

        pos_values = value * util.cast((value > 0), int)
        neg_values = value * util.cast((value < 0), int)
        shift = 2.0 ** (util.round(base))
        # hopefully this elementwise multiplication is broadcasting ?
        value = pos_values * shift * scale - neg_values * shift * scale
        return value

    def _apply(self, value):
        import pdb; pdb.set_trace()
        self._parameter_config = {
            'scale': {
                'initial': tf.ones_initializer(),
                # a vector that has length matches the number of output channels
                # TODO: fix this
                'shape': value.shape,
            }
        }
        return self._quantize(value)
