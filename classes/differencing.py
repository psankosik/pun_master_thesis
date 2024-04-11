import numpy as np

from statsmodels.tsa.stattools import adfuller

def difference(x: np.ndarray, interval=1) -> np.ndarray:
    # List Comprehension version
    return [x[i] - x[i - interval] for i in range(interval, len(x))]

    # Normal For-Loop
    # diff = list()
    # for i in range(interval, len(x)):
    #     value = x[i] - x[i - interval]
    #     diff.append(value)
    # return diff


def apply_difference(dataset: np.ndarray) -> np.ndarray:
    # List Comprehension version
    return np.array([difference(data) for data in dataset])

    # Normal For-Loop
    # results = np.zeros((dataset.shape[0], dataset.shape[1]-1))
    # for i in range(dataset.shape[0]):
    #     results[i,:] = difference(dataset[i])
    # return results


def is_stationary_test(data: np.ndarray) -> bool:
    p_value = adfuller(data)[1]
    if p_value < 0.05:
        # The p-value obtained is less than the significance level of 0.05. So, the time series is, in fact, stationary.
        return True
    else:
        return False
