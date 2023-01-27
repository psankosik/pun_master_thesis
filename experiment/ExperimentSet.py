from experiment.Experiment import Experiment
from utils.hyperparameter import get_successors

class ExperimentSet:
    def __init__(self, classier, datasets=None, augments=None, verbose=0):
        self.classier = classier
        self.datasets = datasets
        self.augments = augments

    def run_all(self):
        # TODO: Refactor the if else
        if len(list(self.augments['params'])) == 1:
            key = list(self.augments['params'])[0]
            params = self.augments['params'][key]
            params_grid = []
            for i in params:
                for dataset in self.datasets:
                    augment = {'name': self.augments['name'], 'function': self.augments['function'], 'enter_label': self.augments['enter_label'], 'params': {key: i}}
                    experiments = Experiment(self.classier, dataset, augment)
                    experiments.run_all()
        else:
            params_grid = get_successors(self.augments['params'])
            for param in params_grid:
                result = {}
                for d in param:
                    result.update(d)

                for dataset in self.datasets:
                    augment = {'name': self.augments['name'], 'function': self.augments['function'], 'enter_label': self.augments['enter_label'], 'params': result}
                    experiments = Experiment(self.classier, dataset, augment)
                    experiments.run_all()
