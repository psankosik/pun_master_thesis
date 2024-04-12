import numpy as np

# ==================== DECOMPOSITION ====================
from statsmodels.tsa.holtwinters import Holt


def apply_double_expo_smoothing(
    dataset: np.ndarray, alpha: float = 0.5, beta: float = 0.5
) -> np.ndarray:
    def double_expo_smoothing(series, alpha: float = 0.5, beta: float = 0.5):
        # Smoothing parameter for level (0 < alpha <= 1)
        # Smoothing parameter for trend (0 < beta <= 1)
        model = Holt(series)
        fit_model = model.fit(
            smoothing_level=alpha, smoothing_trend=beta, optimized=False
        )
        smoothed_values = fit_model.fittedvalues
        return smoothed_values

    return np.array([double_expo_smoothing(data, alpha, beta) for data in dataset])


# ======================= Pattern Mixing — Magnitude ===========================
from imblearn.over_sampling import SMOTE

def smote(
    X: np.ndarray, y: np.ndarray, sampling_strategy: str = 'auto', k_neighbors: int = 5
) -> np.ndarray:
    
    # Determine k_nn
    unique_values, counts = np.unique(y, return_counts=True)
    min_value = min(dict(zip(unique_values, counts)).values())

    # print('B', k_neighbors)
    k_neighbors = min([min_value, k_neighbors])-1
    k_neighbors = 1 if k_neighbors == 0 else k_neighbors
    # print('A', k_neighbors)

    # print(sampling_strategy, k_neighbors)
    smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled


# ========================================================================

# These augmentation function don't need to reshape before augmented
from classes.augmentation.frequency import (
    fourier_transform,
    short_time_fourier_transform,
    wavelet_transform,
)

DONT_NEED_RESHAPE = [
    apply_double_expo_smoothing.__name__,
    smote.__name__,
    fourier_transform.__name__,
    short_time_fourier_transform.__name__,
    wavelet_transform.__name__
]
