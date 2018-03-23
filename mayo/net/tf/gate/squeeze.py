from mayo.net.tf.gate.sparse import SparseRegularizedGatedConvolutionBase


class SqueezeExciteGatedConvolution(SparseRegularizedGatedConvolutionBase):
    def activate(self, tensor):
        tensor = super().activate(tensor)
        return self.actives() * self.gate() * tensor
