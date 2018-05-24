from mayo.session.base import SessionBase


class Test(SessionBase):
    mode = 'validate'

    def __init__(self, config):
        super().__init__(config)
        self.load_checkpoint(self.config.system.checkpoint.load)

    def test(self):
        todo = list(zip(self.task.names, self.task.predictions))
        results = self.run(todo)
        for names, predictions in results:
            self.task.test(names, predictions)
