import numpy as np

from mayo.session.base import SessionBase


class Test(SessionBase):
    mode = 'validate'
    test = True

    def test(self):
        inputs, predictions = self.run(
            [self.task.inputs, self.task.predictions])
        inputs = np.concatenate(inputs, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        for i in range(inputs.shape[0]):
            self.task.test(inputs[i, ...], predictions[i, ...])
