import numpy as np
import pandas as pd

from mlflow import MlflowClient

def query_runs(filter_string: str, order_by: str = "params.dataset ASC"):
    client = MlflowClient()
    runs = client.search_runs("0", filter_string=filter_string, order_by=[order_by])
    return runs


def query_augmented_result(augment_name, augment_params):
    parms_keys = list(augment_params.keys())

    augment = {
        "name": augment_name,
        "param_name": param_name,
        "param_value": param_value,
    }
    filter_string = (
        "params.augmentation LIKE"
        + '"%'
        + f'\'name\': \'{augment["name"]}\', '
        + f'\'params\': {{\'{augment["param_name"]}\': {augment["param_value"]}}}'
        + '%"'
    )
    runs = query_runs(filter_string)
    return runs


def query_baseline(dataset: list, model: str = "minirocket"):
    result = []
    runs = query_runs(f'params.augmentation = "None" AND params.model = "{model}"')
    for i in runs:
        if i.data.params["dataset"] in dataset:
            result.append("{:.3f}".format(i.data.metrics["accuracy"]))
    return result


def compare_result(query_list: list):
    dataset_keys = []
    result_dict = {}

    for i in query_list:
        query_result = query_augmented_result(
            i["augment_name"], i["augment_params"]
        )
        runs = []
        for j in query_result:
            if j.data.params["dataset"] not in dataset_keys:
                dataset_keys.append(j.data.params["dataset"])
            runs.append("{:.3f}".format(j.data.metrics["accuracy"]))
        result_dict[f"{i['param_name']}={i['param_value']}"] = runs

    result_dict["baseline_result"] = query_baseline(dataset_keys)
    df = pd.DataFrame(result_dict, index=dataset_keys)
    def highlight_max(s, props=''):
        return np.where(s == np.nanmax(s.values), props, '')

    df_t = df.T
    df_t = df_t.style.apply(highlight_max, props='color:white;background-color:darkblue', axis=0)
    return df_t



# def query_runs(filter_string: str, order_by: str = "params.dataset ASC"):
#     client = MlflowClient()
#     runs = client.search_runs("0", filter_string=filter_string, order_by=[order_by])
#     return runs


# def query_augmented_result(augment_name, param_name, param_value):
#     augment = {
#         "name": augment_name,
#         "param_name": param_name,
#         "param_value": param_value,
#     }
#     filter_string = (
#         "params.augmentation LIKE"
#         + '"%'
#         + f'\'name\': \'{augment["name"]}\', '
#         + f'\'params\': {{\'{augment["param_name"]}\': {augment["param_value"]}}}'
#         + '%"'
#     )
#     runs = query_runs(filter_string)
#     return runs


# def query_baseline(dataset: list, model: str = "minirocket"):
#     result = []
#     runs = query_runs(f'params.augmentation = "None" AND params.model = "{model}"')
#     for i in runs:
#         if i.data.params["dataset"] in dataset:
#             result.append("{:.3f}".format(i.data.metrics["accuracy"]))
#     return result


# def compare_result(query_list: list):
#     dataset_keys = []
#     result_dict = {}

#     for i in query_list:
#         query_result = query_augmented_result(
#             i["augment_name"], i["param_name"], i["param_value"]
#         )
#         runs = []
#         for j in query_result:
#             if j.data.params["dataset"] not in dataset_keys:
#                 dataset_keys.append(j.data.params["dataset"])
#             runs.append("{:.3f}".format(j.data.metrics["accuracy"]))
#         result_dict[f"{i['param_name']}={i['param_value']}"] = runs

#     result_dict["baseline_result"] = query_baseline(dataset_keys)
#     df = pd.DataFrame(result_dict, index=dataset_keys)
#     def highlight_max(s, props=''):
#         return np.where(s == np.nanmax(s.values), props, '')

#     df_t = df.T
#     df_t = df_t.style.apply(highlight_max, props='color:white;background-color:darkblue', axis=0)
#     return df_t


