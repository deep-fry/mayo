import os

import tensorflow as tf

from mayo.util import memoize_property
from mayo.task.base import TFTaskBase
from mayo.task.image.generate import Preprocess


class ImageTaskBase(TFTaskBase):
    def __init__(self, session, preprocess, shape, moment=None):
        system = session.config.system
        mode = session.mode
        files = session.config.data_files(mode)
        self._preprocessor = Preprocess(
            system, mode, files, preprocess, shape, moment)
        super().__init__(session)

    def augment(self, folder):
        def feed(name):
            image_string = tf.read_file(name)
            image = tf.image.decode_image(image_string)
            image = self._preprocessor.augment(image)
            return name, image
        num_gpus = self.config.system.num_gpus
        batch_size = self.config.system.batch_size_per_gpu * num_gpus
        filenames = tf.constant(os.listdir(folder))
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(feed)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        names, images = iterator.get_next()
        iterer = zip(tf.split(names, num_gpus), tf.split(images, num_gpus))
        for names, images in iterer:
            yield {'input': images}, names

    def generate(self):
        for images, labels in self._preprocessor.preprocess():
            yield {'input': images}, labels

    @memoize_property
    def class_names(self):
        labels_file = self.session.config.dataset.path.label
        with open(labels_file, 'r') as f:
            return f.readlines()
