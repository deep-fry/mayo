import os

from mayo.util import memoize_property
from mayo.task.base import TFTaskBase
from mayo.task.image.generate import Preprocess


class ImageTaskBase(TFTaskBase):
    _truth_keys = NotImplemented

    def __init__(
            self, session, preprocess, num_classes, background_class,
            shape, moment=None):
        bg = background_class
        self.label_offset = int(bg.get('use', 0)) - int(bg.get('has', 0))
        self.num_classes = num_classes + self.label_offset
        session.config.dataset.task.num_classes = self.num_classes
        system = session.config.system
        mode = session.mode
        if mode == 'test':
            files = self._test_files(system.search_path.run.inputs[0])
        else:
            files = session.config.data_files(mode)
        after_shape = preprocess['shape']
        self._preprocessor = Preprocess(
            system, mode, self._truth_keys, files,
            preprocess, shape, after_shape, moment)
        super().__init__(session)

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
