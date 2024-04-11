import pywt
import numpy as np
from scipy.signal import stft


# ======================= Random Transformation — Frequency ===========================


def fourier_transform(
    dataset: np.ndarray, sampling_rate_mul: int, N: int
) -> np.ndarray:
    def apply_fourier_transform(series: np.ndarray, sampling_rate_mul: int, N: int):
        frequencies, amplitudes = np.fft.fftfreq(len(series)), np.abs(
            np.fft.fft(series)
        )
        f_max = frequencies[np.argmax(amplitudes)]

        # Determine sampling_rate (choose a value higher than 2*f_max)
        sampling_rate = sampling_rate_mul * f_max

        # Perform Fourier Transform
        transform_result = np.fft.fft(series, n=N)
        frequencies = np.fft.fftfreq(N, d=1 / sampling_rate)

        # Extract features
        magnitude_spectrum = np.abs(transform_result)
        phase_spectrum = np.angle(transform_result)

        # # Feature extraction
        # median_frequency = np.median(np.abs(frequencies))
        # std_magnitude = np.std(magnitude_spectrum)
        # amplitude = np.max(magnitude_spectrum)
        # dominant_frequency = np.abs(frequencies[np.argmax(magnitude_spectrum)])
        # dominant_phase = phase_spectrum[np.argmax(magnitude_spectrum)]

        return magnitude_spectrum

    return np.array(
        [apply_fourier_transform(data, sampling_rate_mul, N) for data in dataset]
    )


def short_time_fourier_transform(
    dataset: np.ndarray, nperseg: int, noverlap_ratio: float
) -> np.ndarray:
    def apply_stft(series: np.ndarray, nperseg: int, noverlap_ratio: float):
        input_length = len(series)
        if nperseg > input_length:
            nperseg = input_length

        noverlap = nperseg * noverlap_ratio

        frequencies, times, stft_data = stft(
            series, fs=1.0, nperseg=nperseg, noverlap=noverlap
        )

        # Flatten the 2D array to a 1D array
        flattened_stft = np.abs(stft_data).flatten()

        return flattened_stft

    return np.array([apply_stft(data, nperseg, noverlap_ratio) for data in dataset])


def wavelet_transform(dataset: np.ndarray, wavelet_type: str, scale: int) -> np.ndarray:
    def apply_wavelet(series: np.ndarray, wavelet_type: str, scale: int) -> np.ndarray:
        # Perform Continuous Wavelet Transform (CWT)
        scales = np.arange(1, scale)  # Adjust the range of scales based on your data

        coefficients, frequencies = pywt.cwt(series, scales, wavelet_type)

        # Flatten the 2D array to a 1D array
        flattened_coefficients = np.abs(coefficients).flatten()
        return flattened_coefficients

    return np.array([apply_wavelet(data, wavelet_type, scale) for data in dataset])
