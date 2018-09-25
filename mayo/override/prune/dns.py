from mayo.override import util
from mayo.override.base import Parameter
from mayo.override.prune.base import PrunerBase


class MeanStdPruner(PrunerBase):
    alpha = Parameter('alpha', -2, [], 'float')

    def __init__(self, session, alpha=None, should_update=True):
        super().__init__(session, should_update)
        self.alpha = alpha

    def _threshold(self, tensor, alpha=None):
        # axes = list(range(len(tensor.get_shape()) - 1))
        tensor_shape = util.get_shape(tensor)
        axes = list(range(len(tensor_shape)))
        mean, var = util.moments(util.abs(tensor), axes)
        if alpha is None:
            return mean + self.alpha * util.sqrt(var)
        return mean + alpha * util.sqrt(var)

    def _updated_mask(self, var, mask):
        return util.abs(var) > self._threshold(var)

    def _info(self):
        _, mask, density, count = super()._info()
        alpha = self.session.run(self.alpha)
        return self._info_tuple(
            mask=mask, alpha=alpha, density=density, count_=count)


class DynamicNetworkSurgeryPruner(MeanStdPruner):
    """
    References:
        1. https://github.com/yiwenguo/Dynamic-Network-Surgery
        2. https://arxiv.org/abs/1608.04493
    """
    def __init__(
            self, session, alpha=None, on_factor=1.1, off_factor=0.9,
            should_update=True):
        super().__init__(session, alpha, should_update)
        self.on_factor = on_factor
        self.off_factor = off_factor

    def _updated_mask(self, var, mask):
        var, mask, alpha = self.session.run([var, mask, self.alpha])
        threshold = self._threshold(var, alpha)
        on_mask = util.abs(var) > self.on_factor * threshold
        mask = util.logical_or(mask, on_mask)
        off_mask = util.abs(var) > self.off_factor * threshold
        # import pdb; pdb.set_trace()
        return util.logical_and(mask, off_mask)
