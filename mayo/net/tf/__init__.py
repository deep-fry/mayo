from mayo.net.tf.layers import Layers
from mayo.net.tf.gate import GateLayers


class TFNet(Layers, GateLayers):
    """ A class to collate all layer instantiation mixins.  """


__all__ = [TFNet]
