import os

import tensorflow as tf

from mayo.util import memoize_property
from mayo.task.base import TFTaskBase
from mayo.task.image.generate import Preprocess


class ImageTaskBase(TFTaskBase):
    _truth_keys = NotImplemented

    def __init__(self, session, preprocess, shape, moment=None):
        system = session.config.system
        mode = session.mode
        files = session.config.data_files(mode)
        after_shape = preprocess['shape']
        self._preprocessor = Preprocess(
            system, mode, self._truth_keys, files,
            preprocess, shape, after_shape, moment)
        super().__init__(session)

    def augment(self, folder):
        # FIXME this method is somewhat redundant, as similar
        # dataset handling happens in generate.py.
        def feed(name):
            image_string = tf.read_file(name)
            image = tf.image.decode_jpeg(
                image_string, channels=self._preprocessor.shape[-1])
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = self._preprocessor.augment(image)
            return name, image
        num_gpus = self.config.system.num_gpus
        batch_size = self.config.system.batch_size_per_gpu * num_gpus
        filenames = [
            os.path.join(folder, name) for name in sorted(os.listdir(folder))]
        filenames = tf.constant(filenames)
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(feed)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        names, images = iterator.get_next()
        iterer = zip(tf.split(names, num_gpus), tf.split(images, num_gpus))
        for names, images in iterer:
            yield {'input': images}, names

    def generate(self):
        for images, *truths in self._preprocessor.preprocess():
            yield {'input': images}, truths

    def transform(self, net, data, prediction, truth):
        return data['input'], prediction['output'], truth

    @memoize_property
    def class_names(self):
        file = self.session.config.dataset.path.labels
        file = os.path.join('datasets', file)
        with open(file, 'r') as f:
            return f.read().split('\n')
