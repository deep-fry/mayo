import matplotlib
matplotlib.use
import matplotlib.pyplot as plt
import numpy as np
import re
import os
plt.style.use('ggplot')


class GraphPlot(object):
    def __init__(self, net, layers=None):
        super().__init__()
        self.layers = layers
        self.net = net
        self._dir = './plots/'

    def plot(self, session, overrider=None, weights=False, fmaps=False):
        for key, tensor in self.net.layers().items():
            name = self._name_match(key, self.layers)
            if name is not None:
                fmaps = session.run(tensor)
                self._plot_fmaps(fmaps, name)

    def _plot_weights(self, raw_data):
        pass

    def _plot_fmaps(self, raw_data, name, batch=0, style='image'):
        if style == 'image':
            if len(raw_data.shape) != 4:
                raise ValueError(
                    'targeted layer does not have feature maps for plotting!')
            batches, h, w, channels = raw_data.shape
            raw_data = raw_data[batch]
            #check dir
            if not os.path.exists(self._dir):
                os.makedirs(self._dir)
            #plot fmaps
            for data, name in self._prepare_image_plot(raw_data, name):
                matplotlib.image.imsave(name+'.png', data, cmap='gray')

    def _prepare_image_plot(self, raw_data, name):
        for i in range(raw_data.shape[2]):
            image = raw_data[:, :, i]
            image = image / float(np.max(image)) * 255.0
            yield (image, self._dir+'/'+name+'_fmap'+str(i))


    def _polt_image(self, raw_data):
        pass

    def _polt_histogram(self, raw_data):
        pass

    def _name_match(self, name, partial_names):
        for pname in partial_names:
            if pname in name:
                self._dir += re.split(r'/', name)[-2]
                return pname
        return None
