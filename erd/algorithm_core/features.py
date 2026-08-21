"""Feature extraction for the rule-based hierarchical algorithm."""

# 计算功率、ERD、动作分数、左右分数
# Bandpower、ERD、Action Score

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .constants import (
    ACTION_CHANNELS,
    LATERALITY_CHANNELS,
    LEFT_ACTION_WEIGHTS,
    RIGHT_ACTION_WEIGHTS,
    WINDOW_END_S,
    WINDOW_START_S,
)
from .exceptions import InvalidSamplingRateError, InvalidTrialShapeError, MissingChannelError, NonFiniteDataError
from .models import TrialFeatures, TrialInput


def _as_float(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise NonFiniteDataError(f"{name} must be finite.")
    return result


def _channel_index(channel_names: Sequence[str], channel: str) -> int:
    names = [str(name) for name in channel_names]
    if channel not in names:
        raise MissingChannelError(f"Missing required channel: {channel}")
    return names.index(channel)


def _validate_trial(trial: TrialInput) -> np.ndarray:
    data = np.asarray(trial.data, dtype=np.float64)
    if data.ndim != 2:
        raise InvalidTrialShapeError(f"trial data must have shape [channels, samples], got {data.shape}.")
    if data.shape[0] != len(tuple(trial.channel_names)):
        raise InvalidTrialShapeError("trial channel dimension must match channel_names length.")
    sfreq = float(trial.sampling_rate)
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise InvalidSamplingRateError(f"sampling_rate must be positive and finite, got {trial.sampling_rate}.")
    if not np.isfinite(data).all():
        raise NonFiniteDataError("trial data contains NaN or Inf.")
    return data

def _analysis_window_bounds(*, sampling_rate: float, sample_count: int) -> tuple[int, int]:
    rate = _as_float(float(sampling_rate), "sampling_rate")
    if rate <= 0:
        raise InvalidSamplingRateError(f"sampling_rate must be positive and finite, got {sampling_rate}.")
    start = int(round(WINDOW_START_S * rate))
    stop = int(round(WINDOW_END_S * rate))
    if start < 0 or stop <= start or stop > int(sample_count):
        raise InvalidTrialShapeError(
            f"analysis window [{start}, {stop}) exceeds trial samples [0, {sample_count})."
        )
    return start, stop


def compute_mean_square_power_from_preprocessed_trial(
    trial: TrialInput,
    *,
    channels: Sequence[str] = ACTION_CHANNELS,
) -> dict[str, float]:
    """Compute per-channel mean(x^2) from an already 8-30 Hz preprocessed trial."""
    data = _validate_trial(trial)
    start, stop = _analysis_window_bounds(
        sampling_rate=float(trial.sampling_rate),
        sample_count=int(data.shape[-1]),
    )
    powers: dict[str, float] = {}
    for channel in tuple(channels):
        idx = _channel_index(trial.channel_names, channel)
        segment = np.asarray(data[idx, start:stop], dtype=np.float64)
        if segment.size == 0 or not np.isfinite(segment).all():
            raise NonFiniteDataError(f"analysis samples for {channel} must be finite and nonempty.")
        powers[channel] = _as_float(float(np.mean(np.square(segment))), f"power[{channel}]")
    return powers


# 对每个动作通道分别计算功率；输入 trial 已经完成 50 Hz notch + 8–30 Hz bandpass。
def compute_trial_channel_powers(
    trial: TrialInput,
    *,
    channels: Sequence[str] = ACTION_CHANNELS,
) -> dict[str, float]:
    """Compute per-channel 8-30 Hz mean-square power in the fixed 0.5-2.5 s window."""
    return compute_mean_square_power_from_preprocessed_trial(trial, channels=channels)


def compute_dataset_channel_power(dataset: object, channels: Sequence[str]) -> np.ndarray:
    """Compute per-channel powers for a loaded RuleDataset-like object."""
    names = [str(name) for name in getattr(dataset, "ch_names")]
    values = np.asarray(getattr(dataset, "X"), dtype=np.float64)
    if values.ndim != 3:
        raise InvalidTrialShapeError(f"dataset.X must have shape [trials, channels, samples], got {values.shape}.")
    if values.shape[1] != len(names):
        raise InvalidTrialShapeError("dataset channel dimension must match ch_names length.")
    if not np.isfinite(values).all():
        raise NonFiniteDataError("dataset.X contains NaN or Inf.")
    start, stop = _analysis_window_bounds(
        sampling_rate=float(getattr(dataset, "sfreq")),
        sample_count=int(values.shape[-1]),
    )
    powers: list[np.ndarray] = []
    for channel in tuple(channels):
        idx = _channel_index(names, channel)
        powers.append(np.mean(np.square(values[:, idx, start:stop]), axis=-1))
    return np.stack(powers, axis=1).astype(np.float64)

# 对 10 个静息 trial 的同一通道功率取中位数。
def build_rest_baselines(
    rest_trials: Sequence[TrialInput],
    *,
    channels: Sequence[str] = ACTION_CHANNELS,
) -> dict[str, float]:
    """Build channel-wise IQR-cleaned median power baselines from rest calibration trials."""
    details = build_rest_baseline_details(rest_trials, channels=channels)
    return dict(details["baselines"])


def filter_rest_powers_by_iqr(
    *,
    trial_ids: Sequence[str],
    powers: Sequence[float],
) -> dict[str, object]:
    """Apply one-pass per-channel Tukey 1.5 IQR filtering to accepted Rest powers."""
    ids = tuple(str(trial_id) for trial_id in trial_ids)
    values = np.asarray(powers, dtype=np.float64)  # [N], float64
    if values.ndim != 1 or values.size == 0:
        raise NonFiniteDataError("Rest power values for IQR filtering must be a nonempty 1-D array.")
    if len(ids) != int(values.size):
        raise InvalidTrialShapeError("trial_ids length must match Rest power count for IQR filtering.")
    if not np.isfinite(values).all():
        raise NonFiniteDataError("Rest power values for IQR filtering must be finite.")

    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = float(q3 - q1)
    lower_fence = float(q1 - 1.5 * iqr)
    upper_fence = float(q3 + 1.5 * iqr)
    keep_mask = (values >= lower_fence) & (values <= upper_fence)  # [N], bool
    clean_values = values[keep_mask]  # [M<=N], float64
    if clean_values.size == 0 or not np.isfinite(clean_values).all():
        raise NonFiniteDataError("IQR filtering removed all Rest power values.")

    outliers: list[dict[str, object]] = []
    for trial_id, value, keep in zip(ids, values, keep_mask):
        if bool(keep):
            continue
        reason = "below_lower_fence" if float(value) < lower_fence else "above_upper_fence"
        outliers.append(
            {
                "trial_id": str(trial_id),
                "power": float(value),
                "reason": reason,
            }
        )

    return {
        "method": "tukey_1.5_iqr",
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_fence": float(lower_fence),
        "upper_fence": float(upper_fence),
        "outliers": outliers,
        "clean_powers": [float(value) for value in clean_values],
        "clean_sorted_powers": [float(value) for value in np.sort(clean_values)],
        "clean_power_count": int(clean_values.size),
    }


def build_rest_baseline_details(
    rest_trials: Sequence[TrialInput],
    *,
    channels: Sequence[str] = ACTION_CHANNELS,
) -> dict[str, object]:
    """Build IQR-cleaned median baselines and keep raw and clean source powers."""
    rest = tuple(rest_trials)
    if not rest:
        raise InvalidTrialShapeError("rest_trials must not be empty.")
    selected_channels = tuple(str(channel) for channel in channels)
    rows = [compute_trial_channel_powers(trial, channels=selected_channels) for trial in rest]
    trial_ids = tuple(str(trial.trial_id) for trial in rest)
    baselines: dict[str, float] = {}
    channel_details: dict[str, dict[str, object]] = {}
    for channel in selected_channels:
        values = np.asarray([row[channel] for row in rows], dtype=np.float64)
        if values.size == 0 or not np.isfinite(values).all():
            raise NonFiniteDataError(f"baseline source values for {channel} are not finite.")
        filter_details = filter_rest_powers_by_iqr(trial_ids=trial_ids, powers=values)
        clean_values = np.asarray(filter_details["clean_powers"], dtype=np.float64)
        baseline = float(np.median(clean_values))
        if not np.isfinite(baseline) or baseline <= 0:
            raise NonFiniteDataError(f"baseline for {channel} must be positive and finite.")
        baselines[channel] = baseline
        channel_details[channel] = {
            "source_values": [float(value) for value in values],
            "sorted_values": [float(value) for value in np.sort(values)],
            "iqr_filter": {
                "method": str(filter_details["method"]),
                "q1": float(filter_details["q1"]),
                "q3": float(filter_details["q3"]),
                "iqr": float(filter_details["iqr"]),
                "lower_fence": float(filter_details["lower_fence"]),
                "upper_fence": float(filter_details["upper_fence"]),
                "outliers": list(filter_details["outliers"]),
            },
            "clean_values": [float(value) for value in clean_values],
            "clean_sorted_values": [float(value) for value in np.sort(clean_values)],
            "clean_power_count": int(clean_values.size),
            "baseline": float(baseline),
            "aggregation": "median_after_iqr_filter",
        }
    return {
        "baselines": baselines,
        "power_rows": [dict(row) for row in rows],
        "trial_ids": trial_ids,
        "channels": selected_channels,
        "aggregation": "median_after_iqr_filter",
        "iqr_filter": {"method": "tukey_1.5_iqr", "multiplier": 1.5},
        "channel_details": channel_details,
    }

# ERD = (当前功率 - 静息基线功率) / 静息基线功率 × 100%
def compute_channel_erd(
    channel_powers: dict[str, float],
    channel_baselines: dict[str, float],
) -> dict[str, float]:
    """Compute ERD percent as (trial power - baseline) / baseline * 100."""
    erd: dict[str, float] = {}
    for channel, baseline_value in channel_baselines.items():
        if channel not in channel_powers:
            raise MissingChannelError(f"Missing channel power for {channel}.")
        baseline = _as_float(float(baseline_value), f"baseline[{channel}]")
        power = _as_float(float(channel_powers[channel]), f"power[{channel}]")
        if baseline <= 0:
            raise NonFiniteDataError(f"baseline for {channel} must be positive.")
        erd[channel] = float((power - baseline) / baseline * 100.0)
    return erd

# 动作分数 A = max(left ROI evidence, right ROI evidence)
def _validate_action_channels(channels: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(str(channel) for channel in channels)
    if not selected:
        raise MissingChannelError("No action channels selected.")
    unsupported = [channel for channel in selected if channel not in ACTION_CHANNELS]
    if unsupported:
        raise MissingChannelError(f"Unsupported action channel(s): {unsupported}.")
    return selected


def _compute_weighted_roi_score(
    channel_erd: dict[str, float],
    *,
    channels: Sequence[str],
    weights: dict[str, float],
    roi_name: str,
) -> float:
    return float(
        _compute_weighted_roi_details(
            channel_erd,
            channels=channels,
            weights=weights,
            roi_name=roi_name,
        )["score"]
    )


def _compute_weighted_roi_details(
    channel_erd: dict[str, float],
    *,
    channels: Sequence[str],
    weights: dict[str, float],
    roi_name: str,
) -> dict[str, object]:
    selected = set(channels)
    weighted_sum = 0.0
    valid_weight_sum = 0.0
    terms: list[dict[str, float | str]] = []
    for channel, weight in weights.items():
        if channel not in selected:
            continue
        if channel not in channel_erd:
            raise MissingChannelError(f"Missing ERD for {channel}.")
        value = _as_float(float(channel_erd[channel]), f"erd[{channel}]")
        channel_weight = _as_float(float(weight), f"weight[{channel}]")
        weighted = channel_weight * value
        weighted_sum += weighted
        valid_weight_sum += channel_weight
        terms.append(
            {
                "channel": channel,
                "erd": float(value),
                "weight": float(channel_weight),
                "weighted": float(weighted),
            }
        )
    if valid_weight_sum <= 0:
        raise MissingChannelError(f"No valid channels available for {roi_name} action ROI.")
    return {
        "roi": roi_name,
        "terms": terms,
        "weighted_sum": float(weighted_sum),
        "weight_sum": float(valid_weight_sum),
        "erd_pct": float(weighted_sum / valid_weight_sum),
        "drop_pct": float(-(weighted_sum / valid_weight_sum)),
        "score": float(-(weighted_sum / valid_weight_sum)),
    }


def compute_action_score_details(
    channel_erd: dict[str, float],
    *,
    channels: Sequence[str] = ACTION_CHANNELS,
) -> dict[str, object]:
    """Return Stage-1 action score plus display-only left/right ROI details."""
    selected_channels = _validate_action_channels(channels)
    left_details = _compute_weighted_roi_details(
        channel_erd,
        channels=selected_channels,
        weights=LEFT_ACTION_WEIGHTS,
        roi_name="left",
    )
    right_details = _compute_weighted_roi_details(
        channel_erd,
        channels=selected_channels,
        weights=RIGHT_ACTION_WEIGHTS,
        roi_name="right",
    )
    left_score = float(left_details["score"])
    right_score = float(right_details["score"])
    action_score = float(max(left_score, right_score))
    selected_roi = "left" if left_score >= right_score else "right"
    return {
        "left": left_details,
        "right": right_details,
        "action_score": action_score,
        "selected_roi": selected_roi,
        "selected_roi_drop_pct": action_score,
        "selected_roi_erd_pct": float(left_details["erd_pct"] if selected_roi == "left" else right_details["erd_pct"]),
        "left_roi_erd_pct": float(left_details["erd_pct"]),
        "right_roi_erd_pct": float(right_details["erd_pct"]),
        "left_roi_drop_pct": float(left_details["drop_pct"]),
        "right_roi_drop_pct": float(right_details["drop_pct"]),
        "channels": tuple(selected_channels),
    }


def compute_action_score(
    channel_erd: dict[str, float],
    *,
    channels: Sequence[str] = ACTION_CHANNELS,
) -> float:
    """Return Stage-1 action evidence from symmetric weighted left/right ROIs."""
    details = compute_action_score_details(channel_erd, channels=channels)
    return float(details["action_score"])


def compute_action_score_from_values(
    erd_values: Sequence[float],
    *,
    channels: Sequence[str] = ACTION_CHANNELS,
) -> float:
    """Return the action score for an ERD vector with explicit channel identity."""
    selected_channels = _validate_action_channels(channels)
    values = np.asarray(erd_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise InvalidTrialShapeError("action-score ERD values must be a nonempty one-dimensional vector.")
    if values.size != len(selected_channels):
        raise InvalidTrialShapeError("action-score ERD values length must match channels length.")
    if not np.isfinite(values).all():
        raise NonFiniteDataError("action-score ERD values must be a finite one-dimensional vector.")
    channel_erd = {
        channel: float(value)
        for channel, value in zip(selected_channels, values)
    }
    details = compute_action_score_details(channel_erd, channels=selected_channels)
    return float(details["action_score"])


def compute_laterality_score(channel_erd: dict[str, float]) -> float:
    """Return L=ERD_C3-ERD_C4; larger values predict left."""
    c3, c4 = LATERALITY_CHANNELS
    if c3 not in channel_erd or c4 not in channel_erd:
        raise MissingChannelError("Missing C3 or C4 ERD for laterality score.")
    return float(_as_float(float(channel_erd[c3]), "erd[C3]") - _as_float(float(channel_erd[c4]), "erd[C4]"))


def extract_trial_features(
    *,
    trial: TrialInput,
    channel_baselines: dict[str, float],
    action_channels: Sequence[str] | None = None,
) -> TrialFeatures:
    """Extract fixed bandpower, ERD, action score, and laterality score for one trial."""
    channels = tuple(channel_baselines.keys())
    score_channels = tuple(action_channels) if action_channels is not None else ACTION_CHANNELS
    powers = compute_trial_channel_powers(trial, channels=channels)
    erd = compute_channel_erd(powers, channel_baselines)
    action_score_details = compute_action_score_details(erd, channels=score_channels)
    return TrialFeatures(
        channel_powers=powers,
        channel_erd=erd,
        action_score=float(action_score_details["action_score"]),
        laterality_score=0.0,
        action_score_details=dict(action_score_details),
    )


def action_scores_from_power(
    power: np.ndarray,
    baseline: np.ndarray,
    *,
    channels: Sequence[str] = ACTION_CHANNELS,
) -> np.ndarray:
    """Action scores for precomputed powers shaped [trials, selected channels]."""
    selected_channels = _validate_action_channels(channels)
    values = np.asarray(power, dtype=np.float64)
    base = np.asarray(baseline, dtype=np.float64)
    if (
        values.ndim != 2
        or base.ndim != 1
        or values.shape[1] != base.shape[0]
        or values.shape[1] != len(selected_channels)
    ):
        raise InvalidTrialShapeError("power must be [trials, channels] and baseline must be [channels].")
    if not np.isfinite(values).all() or not np.isfinite(base).all() or np.any(base <= 0):
        raise NonFiniteDataError("power and baseline must be finite, with positive baseline.")
    erd = (values - base[None, :]) / base[None, :] * 100.0
    return np.asarray(
        [
            float(compute_action_score_details(
                {
                    channel: float(value)
                    for channel, value in zip(selected_channels, row)
                },
                channels=selected_channels,
            )["action_score"])
            for row in erd
        ],
        dtype=np.float64,
    )


def laterality_scores_from_power(c3_power: np.ndarray, c4_power: np.ndarray, c3_base: float, c4_base: float) -> np.ndarray:
    """Vectorized laterality scores for precomputed C3/C4 powers."""
    c3 = np.asarray(c3_power, dtype=np.float64)
    c4 = np.asarray(c4_power, dtype=np.float64)
    b3 = _as_float(c3_base, "baseline[C3]")
    b4 = _as_float(c4_base, "baseline[C4]")
    if c3.shape != c4.shape or not np.isfinite(c3).all() or not np.isfinite(c4).all() or b3 <= 0 or b4 <= 0:
        raise NonFiniteDataError("laterality power and baselines must be aligned and finite.")
    erd_c3 = (c3 - b3) / b3 * 100.0
    erd_c4 = (c4 - b4) / b4 * 100.0
    return np.asarray(erd_c3 - erd_c4, dtype=np.float64)
