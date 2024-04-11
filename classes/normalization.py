import numpy as np

def z_normalize(sample: np.ndarray) -> np.ndarray:
    mean = np.mean(sample, axis=0)
    std = np.std(sample, axis=0)
    normalized_dataset = (sample - mean) / std
    return normalized_dataset

def apply_z_normalize(dataset: np.ndarray) -> np.ndarray:
    # List Comprehension version
    return np.array([z_normalize(data) for data in dataset])

    # Normal List
    # results = np.zeros((dataset.shape[0], dataset.shape[1]))
    # for i in range(dataset.shape[0]):
    #     results[i,:] = z_normalize(dataset[i])
    # return results