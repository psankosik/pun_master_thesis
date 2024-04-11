import numpy as np

from statistics import mean


def bit_representation(sample: np.ndarray) -> np.ndarray:
    # List Comprehension version
    return np.array([1 if i > mean(sample) else 0 for i in sample])

    # Normal List
    # result_array = list()
    # sample_mean = mean(x['x_train'][0])
    # for i in x['x_train'][0]:
    #     if i > sample_mean:
    #         result_array.append(1)
    #     else:
    #         result_array.append(0)


def apply_bit_representation(dataset: np.ndarray) -> np.ndarray:
    # List Comprehension version
    return np.array([bit_representation(data) for data in dataset])

    # Normal List
    # results = np.zeros((dataset.shape[0], dataset.shape[1]))
    # for i in range(dataset.shape[0]):
    #     results[i,:] = bit_representation(dataset[i])
    # return results