import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
matplotlib.use
plt.style.use('ggplot')


class GraphPlot(object):
    def __init__(
            self, net, variables, model_name, layers=None,
            parent_dir='./plots/'):
        super().__init__()
        self.layers = layers
        self.net = net
        self.variables = variables
        self._dir = parent_dir + model_name + '/'

    def plot(self, session, batch, overrider=None, fmaps_params=None,
             weights_params=None):
        if fmaps_params is not None:
            for key, tensor in self.net.layers().items():
                name = self._name_match(key, self.layers)
                if name is not None:
                    fmaps = session.run(tensor)
                    self._plot_fmaps(fmaps, name, batch, **fmaps_params)
        if weights_params is not None:
            if overrider is not None:
                for o in self.net.overriders:
                    if not isinstance(tensor, list):
                        tensor = o.after
                        name = self._name_match(o.name, self.layers)
                        is_weights = 'weights' in o.name
                    else:
                        tensor = o[overrider].after
                        name = self._name_match(
                            [t.name for t in o], self.layers)
                        is_weights = True
                    if name is not None and is_weights:
                        weights = session.run(tensor)
                        self._plot_weights(weights, name, **weights_params)
            else:
                for tensor in self.variables:
                    name = self._name_match(tensor.name, self.layers)
                    if name is not None and 'weights' in tensor.name:
                        weights = session.run(tensor)
                        self._plot_weights(weights, name, **weights_params)

    def _plot_weights(self, raw_data, name, non_zeros=True, bins_method='fd'):
        self._plot_histogram(raw_data.flatten(), name, non_zeros, bins_method)

    def _plot_fmaps(self, raw_data, name, batch=0, style='image'):
        if style == 'image':
            if len(raw_data.shape) != 4:
                raise ValueError(
                    'targeted layer does not have feature maps for plotting!')
            batches, h, w, channels = raw_data.shape
            raw_data = raw_data[batch]
            # check dir
            if not os.path.exists(self._dir):
                os.makedirs(self._dir)
            # plot fmaps
            for data, name in self._prepare_image_plot(raw_data, name):
                matplotlib.image.imsave(name + '.png', data, cmap='gray')
        if style == 'histogram':
            raw_data = raw_data[batch]
            self._plot_histogram(raw_data.flatten())

    def _prepare_image_plot(self, raw_data, name):
        for i in range(raw_data.shape[2]):
            image = raw_data[:, :, i]
            image = image / float(np.max(image)) * 255.0
            sub_dir = self._dir + '/' + name
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            yield (image, sub_dir + '/' + 'fmap' + str(i))

    def _plot_histogram(
            self, raw_data, name, non_zeros=True, bins_method='fd'):
        if non_zeros:
            raw_data = raw_data[raw_data != 0]
        fig = plt.figure()
        # hist
        n, bins, patches = plt.hist(raw_data, bins=bins_method)
        sub_dir = self._dir + '/' + name
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        plt.savefig(sub_dir + '/' + 'weights.png')
        plt.close(fig)

    def _range(self, start, end, step):
        while start <= end:
            yield start
            start += step

    def _name_match(self, name, partial_names):
        list_mode = isinstance(name, list)
        for pname in partial_names:
            if pname in name and not list_mode:
                return pname
            if list_mode:
                for n in name:
                    if pname in name:
                        return pname
        return None
