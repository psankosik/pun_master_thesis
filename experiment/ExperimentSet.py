from experiment.Experiment import Experiment
from utils.hyperparameter import get_successors


class ExperimentSet:
    def __init__(
        self,
        classifier,
        datasets=None,
        preprocess={"name": str(None), "params": str(None)},
        augments={"name": str(None), "params": str(None)},
        verbose=0,
        save_result=True,
    ):
        self.classifier = classifier
        self.datasets = datasets
        self.preprocess = preprocess
        self.augments = augments
        self.save_result = save_result

    def run_all(self):
        # TODO: Refactor the if else
        if len(list(self.augments["params"])) == 1:
            key = list(self.augments["params"])[0]
            params = self.augments["params"][key]
            params_grid = []
            for i in params:
                for dataset in self.datasets:
                    augment = {
                        "name": self.augments["name"],
                        "function": self.augments["function"],
                        "enter_label": self.augments["enter_label"],
                        "concat_original": self.augments["concat_original"],
                        "params": {key: i},
                    }
                    experiments = Experiment(
                        self.classifier,
                        dataset,
                        self.preprocess,
                        augment,
                        self.save_result,
                    )
                    experiments.run_all()
        else:
            params_grid = get_successors(self.augments["params"])
            for param in params_grid:
                result = {}
                for d in param:
                    result.update(d)

                for dataset in self.datasets:
                    augment = {
                        "name": self.augments["name"],
                        "function": self.augments["function"],
                        "enter_label": self.augments["enter_label"],
                        "concat_original": self.augments["concat_original"],
                        "params": result,
                    }
                    experiments = Experiment(
                        self.classifier,
                        dataset,
                        self.preprocess,
                        augment,
                        self.save_result,
                    )
                    experiments.run_all()
