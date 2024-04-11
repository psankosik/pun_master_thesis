import mlflow
import shortuuid
import traceback

from datetime import datetime


def dict_to_string(dict):
    new_dict = {}
    for i in list(dict.keys()):
        new_dict[i] = str(dict[i])
    return new_dict


def mlflow_save_result(
    metrics: dict[any, str],
    time_param: dict[any: dict],
    model_param: dict[any, str],
    data_param: dict[any, str],
    pre_param: dict[any, str],
    aug_param: dict[any, str],
    jsons: list[dict] = list(),
):

    run_name = datetime.now().strftime("%d/%m/%Y_%H:%M:%S") + "_" + shortuuid.uuid()
    mlflow.start_run(run_name=run_name, experiment_id="870502346412752581")
    run = mlflow.active_run()

    try:
        mlflow.log_metrics(metrics)
        mlflow.log_params(time_param)
        mlflow.log_params(model_param)
        mlflow.log_params(data_param)
        mlflow.log_params(pre_param)
        mlflow.log_params(aug_param)
        for i in jsons:
            mlflow.log_dict(i["data"], i["file_name"])

    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        mlflow.end_run()
