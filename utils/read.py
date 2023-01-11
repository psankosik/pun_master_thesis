import pandas as pd

def read_UCR_dataset_name(filepath=None):
    df = pd.read_csv('DataSummary.csv')
    return df['Name'].to_list()


def read_UCRArchive(path_prefix, dataset):
    df_train = pd.read_csv(
        f"{path_prefix}/{dataset}/{dataset}_TRAIN.tsv", delimiter="\t", header=None
    )
    df_test = pd.read_csv(
        f"{path_prefix}/{dataset}/{dataset}_TEST.tsv", delimiter="\t", header=None
    )

    x_train = (
        df_train.drop([0], axis=1, inplace=False)
        .to_numpy()
        .reshape((df_train.shape[0], df_train.shape[1] - 1, 1))
    )
    y_train = df_train[[0]].to_numpy().reshape((df_train.shape[0]))

    x_test = (
        df_test.drop([0], axis=1, inplace=False)
        .to_numpy()
        .reshape((df_test.shape[0], df_test.shape[1] - 1, 1))
    )
    y_test = df_test[[0]].to_numpy().reshape((df_test.shape[0]))

    return x_train, y_train, x_test, y_test


def reshape_new_to_old_format(ndarr):
    shape = ndarr.shape
    ndarr = ndarr.reshape((shape[0], shape[1], 1))
    return ndarr


def reshape_old_to_new_format(ndarr):
    shape = ndarr.shape
    ndarr = ndarr.reshape((shape[0], 1, shape[1]))
    return ndarr
