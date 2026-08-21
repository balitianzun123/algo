from __future__ import annotations

# 计算 8–30 Hz 频带功率

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import signal


DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "mu": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "mu_beta": (8.0, 30.0),
}

DEFAULT_MOTOR_ROI: tuple[str, ...] = ("FC3", "FC4", "C3", "C4", "CP3", "CPz", "CP4", "Cz")


def select_roi_indices(ch_names: Sequence[str], roi_channels: Sequence[str]) -> list[int]:
    """Return indices for ROI channels that are present in ch_names."""
    name_to_index = {str(name): index for index, name in enumerate(ch_names)}
    indices = [name_to_index[str(name)] for name in roi_channels if str(name) in name_to_index]
    if not indices:
        raise ValueError(f"None of the ROI channels are present in ch_names: {list(roi_channels)}")
    return indices


def compute_bandpower(
    X: NDArray[np.floating],
    sfreq: float,
    band: tuple[float, float] | list[float],
    *,
    method: str = "filter_mean_square",
    filter_order: int = 4,
    window_s: tuple[float, float] | list[float] | None = None,
    epoch_window_s: tuple[float, float] | list[float] | None = None,
    channel_indices: Sequence[int] | None = None,
) -> NDArray[np.floating]:
    """Compute average bandpower for X shaped [trials, channels, samples]."""
    data = _as_3d(X)
    if channel_indices is not None:
        data = data[:, list(channel_indices), :]
    data = _slice_window(data, sfreq, window_s, epoch_window_s)
    low, high = float(band[0]), float(band[1])
    if method == "filter_mean_square":
        filtered = _bandpass(data, sfreq, low, high, filter_order)
        return np.mean(filtered * filtered, axis=(1, 2))
    if method == "welch_psd":
        nperseg = min(data.shape[-1], int(round(float(sfreq))))
        freqs, psd = signal.welch(data, fs=float(sfreq), nperseg=nperseg, axis=-1)
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            raise ValueError(f"No Welch frequency bins found for band {band}.")
        power = np.trapz(psd[..., mask], freqs[mask], axis=-1)
        return np.mean(power, axis=1)
    raise ValueError(f"Unsupported bandpower method: {method}")


def compute_sliding_bandpower(
    X: NDArray[np.floating],
    sfreq: float,
    band: tuple[float, float] | list[float],
    *,
    detection_window_s: tuple[float, float] | list[float],
    epoch_window_s: tuple[float, float] | list[float],
    window_length_s: float = 0.5,
    step_s: float | None = None,
    method: str = "filter_mean_square",
    filter_order: int = 4,
    channel_indices: Sequence[int] | None = None,
) -> tuple[NDArray[np.floating], list[tuple[float, float]]]:
    """Compute bandpower over fixed sliding windows inside the detection window."""
    step = float(step_s) if step_s is not None else float(window_length_s)
    det_start, det_stop = float(detection_window_s[0]), float(detection_window_s[1])
    windows: list[tuple[float, float]] = []
    start = det_start
    while start + float(window_length_s) <= det_stop + 1e-9:
        stop = start + float(window_length_s)
        windows.append((start, stop))
        start += step
    if not windows:
        windows = [(det_start, det_stop)]

    parts = [
        compute_bandpower(
            X,
            sfreq,
            band,
            method=method,
            filter_order=filter_order,
            window_s=window,
            epoch_window_s=epoch_window_s,
            channel_indices=channel_indices,
        )
        for window in windows
    ]
    return np.stack(parts, axis=1), windows


def _as_3d(X: NDArray[np.floating]) -> NDArray[np.floating]:
    data = np.asarray(X, dtype=np.float64)
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    if data.ndim != 3:
        raise ValueError(f"X must have shape [trials, channels, samples] or [channels, samples], got {data.shape}.")
    if not np.isfinite(data).all():
        raise ValueError("Bandpower input contains NaN or Inf.")
    return data


def _slice_window(
    X: NDArray[np.floating],
    sfreq: float,
    window_s: tuple[float, float] | list[float] | None,
    epoch_window_s: tuple[float, float] | list[float] | None,
) -> NDArray[np.floating]:
    if window_s is None:
        return X
    if epoch_window_s is None:
        epoch_start = 0.0
        epoch_text = "[0, duration]"
    else:
        epoch_start = float(epoch_window_s[0])
        epoch_text = str(list(epoch_window_s))
    start = int(round((float(window_s[0]) - epoch_start) * float(sfreq)))
    stop = int(round((float(window_s[1]) - epoch_start) * float(sfreq)))
    if start < 0 or stop <= start or stop > X.shape[-1]:
        raise ValueError(f"Window {list(window_s)} is outside stored epoch {epoch_text}.")
    return X[:, :, start:stop]


def _bandpass(
    X: NDArray[np.floating],
    sfreq: float,
    low: float,
    high: float,
    filter_order: int,
) -> NDArray[np.floating]:
    nyquist = float(sfreq) / 2.0
    if low <= 0 or high >= nyquist or high <= low:
        raise ValueError(f"Invalid band [{low}, {high}] for sfreq={sfreq}.")
    sos = signal.butter(int(filter_order), [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    try:
        return signal.sosfiltfilt(sos, X, axis=-1)
    except ValueError:
        return signal.sosfilt(sos, X, axis=-1)

