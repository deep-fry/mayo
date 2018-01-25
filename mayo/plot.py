import os
import math

import numpy as np

from mayo.log import log


class Plot(object):
    def __init__(self, session, config):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot
        super().__init__()
        self.pyplot = pyplot
        pyplot.style.use('ggplot')
        self.session = session
        self.config = config
        self.net = session.nets[0]

    @property
    def _path(self):
        return self.config.system.search_path.plot[0]

    def plot(self):
        input_tensor = self.net.inputs()['input']
        label_tensor = self.net.labels()
        layer_tensors = self.net.layers()
        variable_tensors = self.net.variables
        input_image, label, layers, variables = self.session.run(
            [input_tensor, label_tensor, layer_tensors, variable_tensors])
        try:
            if self.config.system.plot.features:
                num = self.config.system.batch_size_per_gpu
                for i in range(num):
                    log.info(
                        '{}% Plotting image #{}...'
                        .format(int(i / num * 100.0), i), update=True)
                    path = os.path.join(self._path, str(i))
                    os.makedirs(path, exist_ok=True)
                    # input image
                    # {root}/{index}/input.{ext}
                    input_path = os.path.join(
                        path, 'input-{}'.format(label[i]))
                    self._plot_rgb_image(input_image[i], input_path)
                    # layer activations
                    for node, value in layers.items():
                        if value.ndim != 4:
                            # value is not a (N x H x W x C) layout
                            continue
                        name = node.formatted_name().replace('/', '-')
                        # root/{index}/{layer_name}.{ext}
                        layer_path = os.path.join(path, name)
                        self._plot_images(value[i], layer_path)
            # overridden variable histogram
            if self.config.system.plot.parameters:
                for node, name_value_map in variables.items():
                    for name, value in name_value_map.items():
                        layer_name = node.formatted_name()
                        log.info(
                            'Plotting parameter {} in layer {}'
                            .format(name, layer_name))
                        name = '{}-{}'.format(layer_name, name)
                        name = name.replace('/', '-')
                        # {root}/{layer_name}-{variable_name}.{ext}
                        var_path = os.path.join(self._path, name)
                        self._plot_histogram(value, var_path)
        except KeyboardInterrupt:
            log.info('Abort.')

    def _plot_rgb_image(self, value, path):
        cmap = 'gray'
        if value.ndim == 3:
            if value.shape[-1] == 1:
                value = value[:, :, 0]
            elif value.shape[-1] == 3:
                cmap = None
        path = '{}.{}'.format(path, 'png')
        log.debug('Saving RGB image at {}...'.format(path))
        min_value = np.min(value)
        value = (value - min_value) / (np.max(value) - min_value)
        self.pyplot.imsave(path, value, cmap=cmap)

    def _plot_images(self, value, path):
        if len(value.shape) != 3:
            raise ValueError(
                'We expect number of dimensions to be 4 for image plotting.')
        height, width, channels = value.shape
        max_value = float(np.max(value))
        # plot fmaps
        grid_size = math.ceil(math.sqrt(channels))
        fig = self.pyplot.figure(figsize=(grid_size, grid_size))
        for i in range(channels):
            ax = fig.add_subplot(grid_size, grid_size, i + 1)
            ax.set_axis_off()
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            fmap = value[:, :, i]
            ax.imshow(fmap * 255.0 / max_value, cmap='gray')
        fig.subplots_adjust(wspace=0.025, hspace=0.005)
        path = '{}.{}'.format(path, 'png')
        log.debug('Saving grid of images at {}...'.format(path))
        fig.savefig(path)
        self.pyplot.close(fig)

    def _plot_histogram(self, value, path):
        fig = self.pyplot.figure()
        # histogram
        n, bins, patches = self.pyplot.hist(value.flatten(), bins='fd')
        path = '{}.{}'.format(path, 'eps')
        log.debug('Saving histogram at {}...'.format(path))
        fig.savefig(path)
        self.pyplot.close(fig)
