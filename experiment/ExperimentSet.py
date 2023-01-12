from experiment.Experiment import Experiment


class ExperimentSet:
    def __init__(self, classier, datasets=None, augments=None, verbose=0):
        self.classier = classier
        self.datasets = datasets
        self.augments = augments

    def run_all(self):
        for i in self.datasets:
            experiments = Experiment(self.classier, i, self.augments)
            experiments.run_all()
