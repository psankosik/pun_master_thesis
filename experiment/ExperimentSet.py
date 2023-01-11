from experiment.Experiment import Experiment

class ExperimentSet:
    def __init__(self, classier, datasets=None, augments=None, verbose=0):
        self.classier = classier
        self.datasets = datasets
        self.augments = augments


    def run_experiments(self):
        for i in self.datasets:
            classifer = self.classier
            dataset = i
            experiments = Experiment(classifer, dataset)
            experiments.run_all()