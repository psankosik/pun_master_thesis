from experiment.Experiment import Experiment
from utils.hyperparameter import get_successors

class ExperimentSet:
    def __init__(self, classier, datasets=None, augments=None, verbose=0):
        self.classier = classier
        self.datasets = datasets
        self.augments = augments

    def run_all(self):
        params_grid = get_successors(self.augments['params'])
        for param in params_grid:
            result = {}
            for d in param:
                result.update(d)

            for dataset in self.datasets:
                augment = {'name': self.augments['name'], 'function': self.augments['function'], 'params': result}
                experiments = Experiment(self.classier, dataset, augment)
                experiments.run_all()
