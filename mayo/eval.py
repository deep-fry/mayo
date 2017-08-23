import tensorflow as tf

from mayo.net import Net
from mayo.preprocess import Preprocess


class Evaluate(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._graph = tf.Graph()
        self._preprocessor = Preprocess(self.config)

    def logits(self, images, labels, reuse):
        self._net = Net(
            self.config, images, labels, graph=self._graph, reuse=reuse)
        return self._net.logits()
