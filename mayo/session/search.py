from mayo.log import log
from mayo.session.train import Train


class Search(Train):
    def search(self):
        # profile training accuracy for a given number of epochs
        self._profile()

    def _profile(self):
        epoch = 0
        self.reset_num_epochs()

        baseline = self.config.search.get('baseline')
        if baseline:
            return baseline

        log.info('Profiling baseline accuracy...')
        total_accuracy = 0
        step = 0
        while epoch < self.config.search.profile_epochs:
            epoch = self.run([self.num_epochs], batch=True)
            total_accuracy += self.estimator.get_value('accuracy', 'train')
            step += 1

        self.baseline = total_accuracy / step
        tolerance = self.config.search.tolerance
        self.tolerable_baseline = self.baseline * (1 - tolerance)
        log.info(
            'Baseline accuracy: {}, tolerable accuracy: .'
            .format(self.baseline, self.tolerable_baseline))

        self.reset_num_epochs()
