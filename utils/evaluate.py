import numpy as np
import pandas as pd

from mlflow import MlflowClient
from utils.read import read_UCR_dataset_name


def query_runs(filter_string: str, order_by: str = "params.dataset ASC"):
    client = MlflowClient()
    runs = client.search_runs("0", filter_string=filter_string, order_by=[order_by])
    return runs


def query_augmented_result(augment_name, augment_params, model_name: str):
    params_keys = list(augment_params.keys())
    filter_string = (
        "params.augmentation LIKE"
        + '"%'
        + f"'name': '{augment_name}', "
        + "'params': {"
    )
    for count, key in enumerate(params_keys):
        filter_string += f"'{key}': {augment_params[key]}"
        if count != len(params_keys) - 1:
            filter_string += ", "

    filter_string += '%"'
    runs = query_runs(filter_string)
    runs = filter_model( model_name, runs)
    return runs


def query_baseline(dataset: list, model: str = "minirocket"):
    result = {}
    runs = query_runs(f'params.augmentation = "None" AND params.model = "{model}"')
    for i in runs:
        current_data = i.data.params["dataset"]
        if current_data in dataset:
            result[current_data] = i.data.metrics["accuracy"]
    return result


def query_hc(datasets: list) -> dict:
    df = pd.read_csv("Data/SOTA-DefaultTrainTest.csv", index_col=0)
    hc2 = df["HC2"]

    result = {}
    for data in datasets:
        result[data] = hc2[data]

    return result


def compare_result(
    query_list: list, model_name: str, compare_with: list = ["baseline"], dataset_selection=3.0
):
    datasets = read_UCR_dataset_name(mySelection=dataset_selection)
    variation_table_index = []

    all_variation_acc_list = []
    for i in query_list:
        # Query each augmented variation result
        query_result = query_augmented_result(i["augment_name"], i["augment_params"], model_name)
        acc = clean(query_result, datasets)

        all_acc = {}
        for data in acc:
            # Append each augmented result to the result_dict
            all_acc[data] = float(acc[data])

        all_variation_acc_list.append(all_acc)
        variation_table_index.append(f"{i['augment_params']}")

    for i in compare_with:
        if i == "baseline":
            all_variation_acc_list.append(query_baseline(datasets))
            variation_table_index.append("baseline")
        if i == "state":
            all_variation_acc_list.append(query_hc(datasets))
            variation_table_index.append("HC2")

    df = pd.DataFrame(all_variation_acc_list, index=variation_table_index).fillna(0.0)
    df = df.sort_index(axis=1)

    def highlight_max(s, props=""):
        return np.where(s == np.nanmax(s.values), props, "")

    # df_t = df.T
    df = df.style.apply(highlight_max, props="background-color:darkblue", axis=0)

    print("Warning: Dataset is accuracy missing")
    print(list(set(datasets).difference(set(list(all_acc.keys())))))

    return df


def clean(query_result: list, datasets: int = 3):
    def filter_dataset(query_result, datasets):
        """Filter result that is in specify dataset.
            Ex. if the `query_result` contain dataset ['A', 'B', 'C'] it will result what intersect with datasets.

        Args:
            query_result (_type_): _description_
            datasets (_type_): _description_

        Returns:
            _type_: _description_
        """
        filtered_runs = []
        for run in query_result:
            if run.data.params["dataset"] in datasets:
                filtered_runs.append(run)
        return filtered_runs

    def clean_duplication(query_result):
        # To implement
        return query_result

    def transfrom(query_result):
        result_dict = {}
        for run in query_result:
            result_dict[run.data.params["dataset"]] = "{:.3f}".format(
                run.data.metrics["accuracy"]
            )
        return result_dict

    def sort_dict(d):
        out = dict()
        for k in sorted(d.keys()):
            if isinstance(d[k], dict):
                out[k] = sort_dict(d[k])
            else:
                out[k] = d[k]
        return out

    filtered_result_list = filter_dataset(query_result, datasets)
    filtered_result_list = clean_duplication(filtered_result_list)
    filtered_result_dict = transfrom(filtered_result_list)
    filtered_result_dict = sort_dict(filtered_result_dict)
    return filtered_result_dict


import pandas as pd


def calculate_diff(
    df: pd.DataFrame, column0: int = 0, column1: int = 1
) -> pd.DataFrame:
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        cols_with_zero = (df == 0).any()
        return df.loc[:, ~cols_with_zero]

    def calculate_diff_percentage(
        df: pd.DataFrame, column0: int = 0, column1: int = 1
    ) -> pd.DataFrame:
        column0_results = df.iloc[column0].tolist()
        column1_results = df.iloc[column1].tolist()

        diff_percentages = []
        compare_results = []
        for i in range(0, len(column0_results)):
            if column0_results[i] > column1_results[i]:
                diff = (
                    (column0_results[i] - column1_results[i]) / column1_results[i] * 100
                )
                result = "Win"
            elif column0_results[i] == column1_results[i]:
                diff = 0
                result = "Tie"
            else:
                diff = (
                    (column1_results[i] - column0_results[i]) / column0_results[i] * 100
                )
                result = "Lose"
            diff_percentages.append(diff)
            compare_results.append(result)

        df.loc[4] = diff_percentages
        df.loc[5] = compare_results
        df.rename(index={4: "Percentage Diff", 5: "Result"}, inplace=True)

        return df

    def print_out_stat(df: pd.DataFrame, column0: int = 0, column1: int = 1):
        diff = df.iloc[column0] - df.iloc[column1]
        more = (diff > 0).sum().sum()
        equal = (diff == 0).sum().sum()
        less = (diff < 0).sum().sum()

        index0_name = df.index[column0]
        index1_name = df.index[column1]

        # print the results
        print(
            f"Number of columns where {index0_name} is greater than {index1_name}: {more}"
        )
        print(
            f"Number of columns where {index0_name} and  {index1_name} are equal: {equal}"
        )
        print(
            f"Number of columns where {index0_name} is less than  {index1_name}: {less}"
        )

        # print the result

        more_cols = diff[diff.gt(0)].index.tolist()
        equal_cols = df.columns[(df.iloc[column0] == df.iloc[column1]).values]
        less_cols = diff[diff.lt(0)].index.tolist()

        # print the results
        print()
        print(
            f"Columns where {index0_name} is greater than  {index1_name}:",
            list(more_cols),
        )
        print(
            f"Columns where {index0_name} and  {index1_name} are equal:",
            list(equal_cols),
        )
        print(
            f"Columns where {index0_name} is less than  {index1_name}:", list(less_cols)
        )

    df_clean = clean(df)
    df = calculate_diff_percentage(df_clean, column0, column1)
    print_out_stat(df_clean, column0, column1)
    return df


def filter_model(model_name: str, mlflow_runs: list):
    """To filter mlflow runs with specify model name

    Args:
        model_name (str): _description_
        mlflow_runs (list): _description_

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    result = []
    for run in mlflow_runs:
        if run.data.params['model'] == model_name:
            result.append(run)

    if result == []:
        raise Exception('Model name not found')
    
    return result

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
