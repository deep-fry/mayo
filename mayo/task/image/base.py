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

    def preprocess(self):
        for images, labels in self._preprocessor.preprocess():
            yield {'input': images}, labels
