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
    model_param: dict[any, str],
    data_param: dict[any, str],
    aug_param: dict[any, str],
    jsons: list[dict]=list(),
):

    run_name = datetime.now().strftime("%d/%m/%Y_%H:%M:%S") + "_" + shortuuid.uuid()
    mlflow.start_run(run_name=run_name)
    run = mlflow.active_run()

    print("Active run_id: {}".format(run.info.run_id))
    try:
        mlflow.log_metrics(metrics)
        mlflow.log_params(model_param)
        mlflow.log_params(data_param)
        mlflow.log_params(aug_param)
        for i in jsons:
            mlflow.log_dict(i["data"], i["file_name"])
        mlflow.end_run()
    except Exception as e:
        mlflow.end_run()
        print(e)
        traceback.print_exc()
