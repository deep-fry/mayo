from mayo.override import util
from mayo.override.base import OverriderBase


class QuantizerBase(OverriderBase):
    def eval(self, attribute):
        if util.is_tensor(attribute):
            return self.session.run(attribute)
        return attribute

    def _quantize(self, value, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _overflow_rate(mask):
        """
        Compute overflow_rate from a given overflow mask.  Here `mask` is a
        boolean tensor where True and False represent the presence and absence
        of overflow repsectively.
        """
        return util.sum(util.cast(mask, int)) / util.count(mask)

    def _apply(self, value):
        return self._quantize(value)


class EmptyQuantizer(QuantizerBase):
    def _quantize(self, value):
        return value
