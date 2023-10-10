import os
import numpy as np

from sktime.datasets import load_UCR_UEA_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from classes.augmentation.augmentation import DONT_NEED_RESHAPE

from utils.mlflow_saving import mlflow_save_result
from utils.read import reshape_new_to_old_format, reshape_old_to_new_format


# TODO: Verbose
# TODO: Feature unit-test: taa
# TODO: Describe Function: Get Experiment class information
class Experiment:
    def __init__(
        self,
        clasifier=None,
        dataset=None,
        preprocess={"name": str(None), "params": str(None)},
        augment={"name": str(None), "params": str(None)},
        save_result=True,
        save_data=False,
        verbose=0,
    ):
        self.clasifier = clasifier
        self.dataset = dataset
        self.preprocess = preprocess
        self.augment = augment
        self.y_pred = None
        self.evaluation_metric = {}
        self.save_result = save_result
        self.save_data = save_data
        self.verbose = verbose

    def load_UCR_dataset(self, dataset: str = None):
        dataset_name = self.dataset if self.dataset else dataset
        x_train, y_train = load_UCR_UEA_dataset(
            dataset_name, split="train", return_X_y=True, return_type="numpy2D"
        )
        x_test, y_test = load_UCR_UEA_dataset(
            dataset_name, split="test", return_X_y=True, return_type="numpy2D"
        )
        # TODO: Data Inconsistency with interface
        self.dataset = {
            "name": dataset_name,
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
        }

    # def compile_preprocess(self):
    #     if self.preprocess == None:
    #         return

    #     if self.preprocess.get("name", None) == None:
    #         return

    #     if self.preprocess.get("params", None) == None:
    #         self.dataset["x_train"] = self.preprocess["function"](
    #             self.dataset["x_train"]
    #         )
    #         self.dataset["x_test"] = self.preprocess["function"](self.dataset["x_test"])
    #     else:
    #         self.dataset["x_train"] = self.preprocess["function"](
    #             self.dataset["x_train"], **self.preprocess["params"]
    #         )
    #         self.dataset["x_test"] = self.preprocess["function"](
    #             self.dataset["x_test"], **self.preprocess["params"]
    #         )

    def apply_augmentation(self) -> None:
        """
        This method applies a given augmentation strategy to the dataset.
        """

        def perform_augmentation(x_train, x_test) -> None:
            """
            Performs the augmentation and returns the augmented data.
            """

            def get_augmented_dataset_filename(dataset_type):
                augmented_dataset_folder = "Data_Augmented"
                augmented_param = (
                    str(self.augment["params"])
                    .replace("{", "(")
                    .replace("}", ")")
                    .replace(":", "_")
                    .replace(" ", "")
                )
                return f"{augmented_dataset_folder}/{self.dataset['name']}_{self.augment['name']}_{str(augmented_param)}_{dataset_type}.npy"

            train_augmented_dataset_filename = get_augmented_dataset_filename("TRAIN")
            test_augmented_dataset_filename = get_augmented_dataset_filename("TEST")

            if os.path.exists(train_augmented_dataset_filename) and os.path.exists(
                test_augmented_dataset_filename
            ):
                # Load the augmented dataset
                x_train_aug = np.load(train_augmented_dataset_filename)
                x_test_aug = np.load(test_augmented_dataset_filename)
            else:
                if self.augment.get("enter_label"):
                    x_train_aug = self.augment["function"](
                        x_train, self.dataset["y_train"], **self.augment["params"]
                    )
                    x_test_aug = self.augment["function"](
                        x_test, self.dataset["y_test"], **self.augment["params"]
                    )
                else:
                    x_train_aug = self.augment["function"](
                        x_train, **self.augment["params"]
                    )
                    x_test_aug = self.augment["function"](
                        x_test, **self.augment["params"]
                    )

                if self.save_data:
                    np.save(train_augmented_dataset_filename, x_train_aug)
                    np.save(test_augmented_dataset_filename, x_test_aug)

            return x_train_aug, x_test_aug

        def concatenate_original_dataset(x_train_aug, x_test_aug):
            """
            Concatenates the original dataset to the augmented one and returns the result.
            """
            # Append Training with the same shape
            if self.augment["function"].__name__ in DONT_NEED_RESHAPE:
                x_train_aug = np.concatenate(
                    (x_train_aug, self.dataset["x_train"]), axis=0
                )
                x_test_aug = np.concatenate(
                    (x_test_aug, self.dataset["x_test"]), axis=0
                )
            else:
                x_train_aug = np.concatenate(
                    (x_train_aug, reshape_new_to_old_format(self.dataset["x_train"])),
                    axis=0,
                )
                x_test_aug = np.concatenate(
                    (x_test_aug, reshape_new_to_old_format(self.dataset["x_test"])),
                    axis=0,
                )

            # Append Label
            self.dataset["y_train"] = np.concatenate(
                (self.dataset["y_train"], self.dataset["y_train"]), axis=0
            )
            self.dataset["y_test"] = np.concatenate(
                (self.dataset["y_test"], self.dataset["y_test"]), axis=0
            )

            return x_train_aug, x_test_aug

        # 0. Augment or not
        if not self.augment["name"] != "None":
            """
            Sets the augmented data to be equal to the original data.
            """
            self.dataset["x_train_aug"] = self.dataset["x_train"]
            self.dataset["x_test_aug"] = self.dataset["x_test"]
            return

        # 1. Reshape or not
        if self.augment["function"].__name__ in DONT_NEED_RESHAPE:
            x_train = self.dataset["x_train"]
            x_test = self.dataset["x_test"]
        else:
            x_train = reshape_new_to_old_format(self.dataset["x_train"])
            x_test = reshape_new_to_old_format(self.dataset["x_test"])

        # 2. Apply augmentation
        x_train_aug, x_test_aug = perform_augmentation(x_train, x_test)

        # 3. Concat original or not
        if self.augment.get("concat_original", False):
            x_train_aug, x_test_aug = concatenate_original_dataset(
                x_train_aug, x_test_aug
            )

        # 4. Assign self.dataset["x_train_aug"] = x_train_aug, self.dataset["x_test_aug"] = x_test_aug
        if self.augment["function"].__name__ in DONT_NEED_RESHAPE:
            self.dataset["x_train_aug"] = x_train_aug
            self.dataset["x_test_aug"] = x_test_aug
        else:
            self.dataset["x_train_aug"] = reshape_old_to_new_format(x_train_aug)
            self.dataset["x_test_aug"] = reshape_old_to_new_format(x_test_aug)

    def train_classier(self):
        if (self.dataset.get("x_train_aug").all() == None) or (
            self.dataset.get("x_test_aug").all() == None
        ):
            self.clasifier["function"].fit(
                self.dataset["x_train"], self.dataset["y_train"]
            )
        else:
            self.clasifier["function"].fit(
                self.dataset["x_train_aug"], self.dataset["y_train"]
            )

    def predict(self):
        self.y_pred = self.clasifier["function"].predict(self.dataset["x_test_aug"])

    def evaluate(self):
        self.evaluation_metric["accuracy"] = accuracy_score(
            self.dataset["y_test"], self.y_pred
        )
        (
            self.evaluation_metric["precision"],
            self.evaluation_metric["recall"],
            self.evaluation_metric["fbeta"],
            self.evaluation_metric["support"],
        ) = precision_recall_fscore_support(self.dataset["y_test"], self.y_pred)

    def save_result_to_mlflow(self):
        if self.save_result:
            mlflow_save_result(
                {"accuracy": self.evaluation_metric["accuracy"]},
                {"model": self.clasifier["name"]},
                {
                    "dataset": self.dataset["name"],
                    "datapoint_shape": str(self.dataset["x_train_aug"].shape)
                    + "x"
                    + str(self.dataset["x_test"].shape),
                    "number_of_class": len(set(list(self.dataset["y_test"]))),
                    "concat_original": self.augment.get("concat_original"),
                },
                {
                    "preprocess": {
                        "name": self.preprocess["name"],
                        "params": self.preprocess.get("params", "None"),
                    },
                },
                {
                    "augmentation": {
                        "name": self.augment["name"],
                        "params": self.augment.get("params", "None"),
                    },
                },
                # [{"data": self.augment["params"], "file_name": "dict/augmentation.json"}],
            )
        print(
            f'{self.clasifier["name"]}, {self.dataset["name"]}, ({self.preprocess["name"]}, {self.preprocess.get("params", "None")}), ({self.augment["name"]}, {self.augment.get("params", "None")}), {self.evaluation_metric["accuracy"]} DONE'
        )

    def run_all(self):
        self.load_UCR_dataset()
        # self.compile_preprocess()
        self.apply_augmentation()
        self.train_classier()
        self.predict()
        self.evaluate()
        self.save_result_to_mlflow()


# {
#     "Metrics": {
#         "augmentation": {
#             "wall": 100,
#             "cpu": 100
#         },
#         "classification": {
#             "wall": 100,
#             "cpu": 100
#         },
#         "prediction": {
#             "wall": 100,
#             "cpu": 100
#         },
#         "total": {
#             "wall": 100,
#             "cpu": 100            
#         }
#     }
# }