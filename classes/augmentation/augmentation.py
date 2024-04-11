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


# These augmentation function don't need to reshape before augmented
from classes.augmentation.frequency import (
    fourier_transform,
    short_time_fourier_transform,
    wavelet_transform,
)

DONT_NEED_RESHAPE = [
    apply_double_expo_smoothing.__name__,
    fourier_transform.__name__,
    short_time_fourier_transform.__name__,
    wavelet_transform.__name__
]
