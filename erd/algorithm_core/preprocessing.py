"""Legacy-compatible EEG preprocessing for dataset construction and product input.

This module intentionally mirrors the preprocessing used to build the frozen
MI-only dataset: channel selection by exact names, trial-level 50 Hz notch,
trial-level 8-30 Hz IIR bandpass, and a 0-4 s stop-exclusive epoch crop.
It does not compute ERD features, thresholds, predictions, or metrics.
"""

# 原始 EEG 的 50 Hz 陷波、8–30 Hz 带通和分段
# notch、bandpass、通道选择、epoch

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .constants import (
    PREPROCESSING_BANDPASS_HZ,
    PREPROCESSING_EPOCH_WINDOW_S,
    PREPROCESSING_FILTER_METHOD,
    PREPROCESSING_IIR_ORDER,
    PREPROCESSING_NOTCH_HZ,
    PREPROCESSING_OUTPUT_DTYPE,
    REQUIRED_CHANNELS,
    SUPPORTED_RAW_DATA_UNITS,
)
from .exceptions import (
    DuplicateChannelError,
    InvalidEventSampleError,
    InvalidSamplingRateError,
    InvalidTrialShapeError,
    MissingChannelError,
    NonFiniteDataError,
    RawTrialTooShortError,
    UnsupportedDataUnitError,
)
from .models import RawTrialInput, TrialInput
from . import trace as eeg_trace


EPOCH_WINDOW_S: tuple[float, float] = PREPROCESSING_EPOCH_WINDOW_S
SUPPORTED_DATA_UNITS: tuple[str, ...] = SUPPORTED_RAW_DATA_UNITS


def channel_indices(all_channels: Sequence[str], selected_channels: Sequence[str]) -> list[int]:
    """Return exact-name indices for selected channels, matching the legacy builder."""
    names = _validate_channel_names(all_channels)
    selected = tuple(str(channel) for channel in selected_channels)
    missing = [channel for channel in selected if channel not in names]
    if missing:
        raise MissingChannelError(f"Missing required channel(s): {missing}")
    name_to_index = {name: index for index, name in enumerate(names)}
    return [name_to_index[channel] for channel in selected]


def validate_raw_trial(trial: RawTrialInput) -> None:
    """Validate a raw trial without modifying its data."""
    data = np.asarray(trial.data)
    if data.ndim != 2:
        raise InvalidTrialShapeError(f"raw trial data must have shape [channels, samples], got {data.shape}.")
    if data.shape[0] != len(tuple(trial.channel_names)):
        raise InvalidTrialShapeError("raw trial channel dimension must match channel_names length.")
    _validate_channel_names(trial.channel_names)
    channel_indices(trial.channel_names, REQUIRED_CHANNELS)
    _validate_sampling_rate(trial.sampling_rate)
    _validate_data_unit(trial.data_unit)
    if not np.isfinite(np.asarray(data, dtype=np.float64)).all():
        raise NonFiniteDataError("raw trial data contains NaN or Inf; legacy dataset builder drops such trials.")
    _validate_event_window(
        n_samples=int(data.shape[1]),
        sampling_rate=float(trial.sampling_rate),
        event_sample=trial.event_sample,
    )


def select_and_reorder_channels(
    *,
    data: np.ndarray,
    channel_names: tuple[str, ...] | Sequence[str],
    selected_channels: tuple[str, ...] | Sequence[str] = REQUIRED_CHANNELS,
) -> np.ndarray:
    """Select target channels by exact name and return them in the configured order."""
    array = np.asarray(data)
    if array.ndim != 2:
        raise InvalidTrialShapeError(f"data must have shape [channels, samples], got {array.shape}.")
    names = _validate_channel_names(channel_names)
    if array.shape[0] != len(names):
        raise InvalidTrialShapeError("data channel dimension must match channel_names length.")
    indices = channel_indices(names, selected_channels)
    return np.asarray(array[indices, :]).copy()


def apply_notch_filter(
    data: np.ndarray | None = None,
    *,
    sampling_rate: float | None = None,
    sfreq: float | None = None,
    notch_freq: float | Sequence[float] | None = PREPROCESSING_NOTCH_HZ,
    method: str = PREPROCESSING_FILTER_METHOD,
    iir_order: int = PREPROCESSING_IIR_ORDER,
) -> np.ndarray:
    """Apply the legacy MNE IIR notch filter on the last axis."""
    array = _validate_filter_array(data)
    rate = _resolve_sampling_rate(sampling_rate=sampling_rate, sfreq=sfreq)
    if notch_freq is None:
        return array
    import mne

    iir_params = {"order": int(iir_order), "ftype": "butter"} if str(method) == "iir" else None
    return np.asarray(
        mne.filter.notch_filter(
            array,
            Fs=float(rate),
            freqs=notch_freq,
            method=str(method),
            iir_params=iir_params,
            verbose=False,
        ),
        dtype=np.float64,
    )


def apply_bandpass_filter(
    data: np.ndarray | None = None,
    *,
    sampling_rate: float | None = None,
    sfreq: float | None = None,
    l_freq: float | None = PREPROCESSING_BANDPASS_HZ[0],
    h_freq: float | None = PREPROCESSING_BANDPASS_HZ[1],
    method: str = PREPROCESSING_FILTER_METHOD,
    iir_order: int = PREPROCESSING_IIR_ORDER,
) -> np.ndarray:
    """Apply the legacy MNE IIR bandpass filter on the last axis."""
    array = _validate_filter_array(data)
    rate = _resolve_sampling_rate(sampling_rate=sampling_rate, sfreq=sfreq)
    if l_freq is None and h_freq is None:
        return array
    import mne

    iir_params = {"order": int(iir_order), "ftype": "butter"} if str(method) == "iir" else None
    return np.asarray(
        mne.filter.filter_data(
            array,
            sfreq=float(rate),
            l_freq=l_freq,
            h_freq=h_freq,
            method=str(method),
            iir_params=iir_params,
            verbose=False,
        ),
        dtype=np.float64,
    )


def extract_event_aligned_epoch(
    *,
    data: np.ndarray,
    sampling_rate: float,
    event_sample: int | None,
    epoch_window_s: tuple[float, float] = EPOCH_WINDOW_S,
) -> np.ndarray:
    """Extract the legacy event-aligned epoch using stop-exclusive samples."""
    array = np.asarray(data)
    if array.ndim != 2:
        raise InvalidTrialShapeError(f"data must have shape [channels, samples], got {array.shape}.")
    start, stop = _epoch_bounds(
        sampling_rate=float(sampling_rate),
        event_sample=event_sample,
        n_samples=int(array.shape[1]),
        epoch_window_s=epoch_window_s,
    )
    return np.asarray(array[:, start:stop]).copy()


def preprocess_raw_trial(
    trial: RawTrialInput,
    *,
    output_dtype: str | np.dtype[Any] = PREPROCESSING_OUTPUT_DTYPE,
) -> TrialInput:
    """Return one legacy-preprocessed 0-4 s trial ready for feature extraction.

    Processing order matches the dataset builder for already trialized raw NPY
    input: validate -> channel select/reorder -> notch -> bandpass -> 0-4 s crop.
    """
    validate_raw_trial(trial)
    selected = select_and_reorder_channels(data=np.asarray(trial.data), channel_names=trial.channel_names)
    notched = apply_notch_filter(data=selected, sampling_rate=float(trial.sampling_rate))
    filtered = apply_bandpass_filter(data=notched, sampling_rate=float(trial.sampling_rate))
    epoch = extract_event_aligned_epoch(
        data=filtered,
        sampling_rate=float(trial.sampling_rate),
        event_sample=trial.event_sample,
    )
    preprocess_summary = eeg_trace.build_preprocess_summary(
        raw_selected=selected,
        filtered_epoch=epoch,
        sampling_rate=float(trial.sampling_rate),
        channel_names=tuple(REQUIRED_CHANNELS),
    )
    eeg_trace.trace_preprocess(
        trial_id=str(trial.trial_id),
        sampling_rate=float(trial.sampling_rate),
        raw_selected=selected,
        filtered_epoch=epoch,
        channel_names=tuple(REQUIRED_CHANNELS),
        notch_hz=PREPROCESSING_NOTCH_HZ,
        bandpass_hz=PREPROCESSING_BANDPASS_HZ,
    )
    return TrialInput(
        trial_id=str(trial.trial_id),
        data=np.asarray(epoch, dtype=np.dtype(output_dtype)),
        channel_names=tuple(REQUIRED_CHANNELS),
        sampling_rate=float(trial.sampling_rate),
        label=trial.label,
        preprocess_summary=dict(preprocess_summary),
    )


def _validate_channel_names(channel_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(name) for name in channel_names)
    if not names:
        raise MissingChannelError("channel_names must not be empty.")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise DuplicateChannelError(f"Duplicate channel name(s): {duplicates}")
    return names


def _validate_sampling_rate(sampling_rate: float) -> float:
    rate = float(sampling_rate)
    if not np.isfinite(rate) or rate <= 0:
        raise InvalidSamplingRateError(f"sampling_rate must be positive and finite, got {sampling_rate}.")
    # 频率窗保持 Hz；样本索引由秒数和实际采样率动态换算。
    if rate <= 100.0:
        raise InvalidSamplingRateError(f"sampling_rate must be greater than 100 Hz for 50 Hz preprocessing, got {rate}.")
    return rate


def _resolve_sampling_rate(*, sampling_rate: float | None, sfreq: float | None) -> float:
    value = sampling_rate if sampling_rate is not None else sfreq
    if value is None:
        raise InvalidSamplingRateError("sampling_rate is required.")
    return _validate_sampling_rate(float(value))


def _validate_data_unit(data_unit: str) -> str:
    unit = str(data_unit)
    if not unit:
        raise UnsupportedDataUnitError("data_unit must be explicit.")
    if unit not in SUPPORTED_DATA_UNITS:
        raise UnsupportedDataUnitError(
            f"unsupported data_unit={unit!r}; expected one of {tuple(SUPPORTED_DATA_UNITS)}."
        )
    return unit


def _validate_filter_array(data: np.ndarray | None) -> np.ndarray:
    if data is None:
        raise InvalidTrialShapeError("data is required.")
    array = np.asarray(data)
    if array.ndim not in (2, 3):
        raise InvalidTrialShapeError(f"filter input must be [channels, samples] or [trials, channels, samples], got {array.shape}.")
    if not np.isfinite(np.asarray(array, dtype=np.float64)).all():
        raise NonFiniteDataError("filter input contains NaN or Inf.")
    return np.asarray(array, dtype=np.float64)


def _validate_event_window(*, n_samples: int, sampling_rate: float, event_sample: int | None) -> None:
    _epoch_bounds(
        sampling_rate=sampling_rate,
        event_sample=event_sample,
        n_samples=n_samples,
        epoch_window_s=EPOCH_WINDOW_S,
    )


def _epoch_bounds(
    *,
    sampling_rate: float,
    event_sample: int | None,
    n_samples: int,
    epoch_window_s: tuple[float, float],
) -> tuple[int, int]:
    rate = _validate_sampling_rate(sampling_rate)
    if event_sample is None:
        event = 0
    elif isinstance(event_sample, (int, np.integer)):
        event = int(event_sample)
    else:
        raise InvalidEventSampleError(f"event_sample must be an integer or None, got {event_sample!r}.")
    if event < 0 or event >= int(n_samples):
        raise InvalidEventSampleError(f"event_sample={event} is outside available samples [0, {n_samples}).")
    start_offset = int(round(float(epoch_window_s[0]) * rate))
    stop_offset = int(round(float(epoch_window_s[1]) * rate))
    if start_offset < 0 or stop_offset <= start_offset:
        raise InvalidEventSampleError(f"invalid epoch_window_s={epoch_window_s}.")
    start = event + start_offset
    stop = event + stop_offset
    if stop > int(n_samples):
        raise RawTrialTooShortError(
            f"event-aligned epoch [{start}, {stop}) exceeds available samples [0, {n_samples})."
        )
    return start, stop
