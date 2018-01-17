import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re
import os
matplotlib.use
plt.style.use('ggplot')


class GraphPlot(object):
    def __init__(self, net, variables, layers=None, parent_dir='./plots/'):
        super().__init__()
        self.layers = layers
        self.net = net
        self.variables = variables
        self._dir = parent_dir

    def plot(self, session, batch, overrider=None, fmaps_params=None,
             weights_params=None):
        if fmaps_params is not None:
            for key, tensor in self.net.layers().items():
                name = self._name_match(key, self.layers)
                if name is not None:
                    fmaps = session.run(tensor)
                    self._plot_fmaps(fmaps, name, batch, **fmaps_params)
                    import pdb; pdb.set_trace()
        if weights_params is not None:
            for tensor in self.net.overriders:
                if overrider:
                    if not isinstance(tensor, list):
                        tensor = tensor.after
                    else:
                        tensor = tensor[overrider].after
                name = self._name_match(tensor.name, self.layers)
                if name is not None:
                    weights = session.run(tensor)
                    self._plot_weights(weights, name, **weights_params)

    def _plot_weights(self, raw_data, name, smooth=True, smooth_scale=20):
        self._plot_histogram(raw_data.flatten(), name, smooth, smooth_scale)

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
            yield (image, self._dir + '/' + name + '_fmap' + str(i))

    def _plot_histogram(self, raw_data, name, smooth=True, smooth_scale=20):
        figure = plt.figure()
        x_axis, y_axis = self._reform(raw_data, smooth, smooth_scale)
        plt.plot(x_axis, y_axis)
        plt.save_fig(name)

    def _reform(self, value, smooth, smooth_scale, scale=1):
        assert isinstance(value, np.ndarray), 'np array is required!'
        x_axis = []
        y_axis = []
        start = int(np.min(value) - scale)
        end = int(np.max(value) + scale)
        for interval in self._range(start, end, 2 * scale):
            x_axis.append(interval)
            indexing = np.logical_and(
                value > interval - scale, value < interval + scale)
            y_axis.append(value[indexing].size)
        x_axis = np.array(x_axis)
        y_axis = np.array(y_axis)
        if smooth:
            x_tmp = np.linspace(
                x_axis.min(), x_axis.max(), x_axis.size * smooth_scale)
            y_axis = spline(x_axis, y_axis, x_tmp)
            x_axis = x_tmp
        return (x_axis, y_axis)

    def _name_match(self, name, partial_names):
        for pname in partial_names:
            if pname in name:
                self._dir += re.split(r'/', name)[-2]
                return pname
        return None
