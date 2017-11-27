from mayo.net.tf.layers import Layers
from mayo.net.tf.gate import GateLayers
from mayo.net.tf.hadamard import HadamardLayers


class TFNet(Layers, GateLayers, HadamardLayers):
    """ A class to collate all layer instantiation mixins.  """


__all__ = [TFNet]
