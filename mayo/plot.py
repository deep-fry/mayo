import os
import math
import numpy as np
from PIL import Image

from mayo.log import log
from mayo.task.image.classify import Classify


class Plot(object):
    def __init__(self, session, config):
        super().__init__()
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot
        pyplot.style.use('ggplot')
        self.pyplot = pyplot
        self.session = session
        self.config = config
        self.task = session.task
        self.net = session.task.nets[0]
        if not isinstance(session.task, Classify):
            raise TypeError('We only support classification task for now.')

    @property
    def _path(self):
        return self.config.system.search_path.plot[0]

    def plot(self):
        input_tensor = self.task.inputs[0]
        label_tensor = self.task.truths[0]
        layer_tensors = self.net.layers()
        variable_tensors = self.net.variables
        input_image, label, layers, variables = self.session.run(
            [input_tensor, label_tensor, layer_tensors, variable_tensors])
        try:
            if self.config.system.plot.get('features'):
                self.plot_features(input_image, label, layers)
            if self.config.system.plot.get('parameters'):
                # overridden variable histogram
                self.plot_parameters(variables)
            if self.config.system.plot.get('gates'):
                self.plot_gate_heatmaps()
        except KeyboardInterrupt:
            log.info('Abort.')

    def plot_features(self, input_image, label, layers):
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

    def plot_parameters(self, variables):
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

    def _plot_histogram(self, value, path):
        fig = self.pyplot.figure()
        # histogram
        n, bins, patches = self.pyplot.hist(value.flatten(), bins='fd')
        path = '{}.{}'.format(path, 'eps')
        log.debug('Saving histogram at {}...'.format(path))
        fig.savefig(path)
        self.pyplot.close(fig)

    def plot_gate_heatmaps(self):
        def path(node, key):
            # {root}/gate/gamma-{node}.{ext}
            node_name = node.formatted_name().replace('/', '-')
            path = 'gate/{}-{}'.format(key, node_name)
            return os.path.join(self._path, path)

        gammas = self.session.estimator.get_histories('gate.gamma')
        actives = self.session.estimator.get_histories('gate.active')
        if not gammas and not actives:
            return
        gamma_heatmaps = self._heatmaps(gammas)
        active_heatmaps = self._heatmaps(actives)

        for node in gamma_heatmaps:
            gamma_path = path(node, 'gamma')
            self._plot_heatmap(gamma_heatmaps[node], gamma_path, vmin=0)
            active_path = path(node, 'active')
            actives = active_heatmaps.get(node)
            if actives is not None:
                self._plot_heatmap(actives, active_path, vmin=0, vmax=1)

    def _heatmaps(self, histories):
        labels_history = self.session.estimator.get_history('truth')
        label_keys = set()
        # collect by node->label->history
        hmap = {}
        for node, history in histories.items():
            lmap = hmap.setdefault(node, {})
            for labels, values in zip(labels_history, history):
                values = np.squeeze(values, [1, 2])
                for label, value in zip(labels, values):
                    label_keys.add(label)
                    lmap.setdefault(label, []).append(value)
        # average history
        for node, lmap in hmap.items():
            values = []
            for label in range(len(label_keys)):
                # labels will be continuous
                values.append(np.mean(lmap[label], axis=0))
            hmap[node] = np.stack(values, axis=0)
        return hmap

    def _plot_heatmap(self, heatmap, path, vmin=None, vmax=None):
        if vmin is None:
            vmin = np.min(heatmap)
        if vmax is None:
            vmax = np.max(heatmap)
        if vmin >= vmax:
            raise ValueError(
                'The minimum value is not less than the maximum value.')
        heatmap = np.uint8((heatmap - vmin) / (vmax - vmin) * 255.0)
        image = Image.fromarray(heatmap)
        path = '{}.{}'.format(path, 'png')
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        log.debug('Saving gates heatmap at {}...'.format(path))
        image.save(path)
