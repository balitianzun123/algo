"""Personal Rest Power baseline calibration."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import trace as eeg_trace
from .constants import (
    ACTION_CHANNELS,
    DEFAULT_REQUIRED_REST_TRIALS,
    LEFT_ID,
    LEFT_LABEL,
    MIN_REQUIRED_REST_TRIALS,
    REST10_LATERALITY_THRESHOLD,
    REST_ID,
    REST_LABEL,
    RIGHT_ID,
    RIGHT_LABEL,
)
from .criterion import (
    MI_SUCCESS_CRITERION_NAME,
    MI_SUCCESS_CRITERION_VERSION,
    get_mi_success_drop_threshold_pct,
)
from .exceptions import InvalidCalibrationCountError, InvalidCalibrationLabelError, InvalidRequiredRestTrialsError
from .features import build_rest_baseline_details, filter_rest_powers_by_iqr
from .models import PersonalProfile, TrialInput


def validate_required_rest_trials(required_rest_trials: object) -> int:
    """Return a valid Rest-N count, rejecting bools and non-integral values."""
    if isinstance(required_rest_trials, (bool, np.bool_)) or not isinstance(required_rest_trials, (int, np.integer)):
        raise InvalidRequiredRestTrialsError("required_rest_trials must be an integer (bool is not accepted).")
    count = int(required_rest_trials)
    if count < MIN_REQUIRED_REST_TRIALS:
        raise InvalidRequiredRestTrialsError(
            f"required_rest_trials must be at least {MIN_REQUIRED_REST_TRIALS}, got {count}."
        )
    return count


def _check_count(trials: Sequence[TrialInput], expected: int, name: str) -> None:
    if len(tuple(trials)) != int(expected):
        raise InvalidCalibrationCountError(f"{name} must contain exactly {expected} trials.")


def _check_labels(trials: Sequence[TrialInput], expected_label: str, name: str) -> None:
    bad = [trial.trial_id for trial in trials if trial.label != expected_label]
    if bad:
        raise InvalidCalibrationLabelError(f"{name} contains trials without label {expected_label}: {bad[:3]}")


def _check_sampling_and_channels(trials: Sequence[TrialInput], channel_names: Sequence[str], sampling_rate: float) -> None:
    expected_channels = tuple(channel_names)
    expected_sfreq = float(sampling_rate)
    for trial in trials:
        if tuple(trial.channel_names) != expected_channels:
            raise InvalidCalibrationLabelError(f"trial {trial.trial_id} channel_names do not match calibration input.")
        if float(trial.sampling_rate) != expected_sfreq:
            raise InvalidCalibrationLabelError(f"trial {trial.trial_id} sampling_rate does not match calibration input.")


def _profile(
    *,
    subject_id: str,
    mode: str,
    sampling_rate: float,
    channel_names: Sequence[str],
    channel_baselines: dict[str, float],
    action_threshold: float,
    laterality_threshold: float,
    threshold_source_action: str,
    threshold_source_laterality: str,
    calibration_trial_ids: Sequence[str],
    action_thresholds_without_channel: dict[str, float] | None = None,
    extra: dict[str, object] | None = None,
) -> PersonalProfile:
    return PersonalProfile(
        profile_id=f"{subject_id}_{mode}",
        subject_id=str(subject_id),
        mode=str(mode),
        sampling_rate=float(sampling_rate),
        channel_names=tuple(str(ch) for ch in channel_names),
        channel_baselines={str(ch): float(value) for ch, value in channel_baselines.items()},
        action_threshold=float(action_threshold),
        laterality_threshold=float(laterality_threshold),
        action_thresholds_without_channel={
            str(ch): float(value)
            for ch, value in dict(action_thresholds_without_channel or {}).items()
        },
        threshold_source_action=str(threshold_source_action),
        threshold_source_laterality=str(threshold_source_laterality),
        calibration_trial_ids=tuple(str(x) for x in calibration_trial_ids),
        extra=dict(extra or {}),
    )


def _power_details_for_profile(baseline_details: dict[str, object]) -> dict[str, dict[str, object]]:
    channel_details = dict(baseline_details.get("channel_details", {}) or {})
    result: dict[str, dict[str, object]] = {}
    for channel in tuple(str(channel) for channel in baseline_details.get("channels", ACTION_CHANNELS)):
        details = dict(channel_details[channel])
        clean_values = list(details.get("clean_values", details.get("source_values", [])))
        result[channel] = {
            "trial_ids": [str(trial_id) for trial_id in baseline_details.get("trial_ids", ())],
            "trial_powers": [float(value) for value in details.get("source_values", [])],
            "sorted_powers": [float(value) for value in details.get("sorted_values", [])],
            "iqr_filter": dict(details.get("iqr_filter", {})),
            "clean_powers": [float(value) for value in clean_values],
            "clean_sorted_powers": [
                float(value)
                for value in details.get("clean_sorted_values", sorted(float(value) for value in clean_values))
            ],
            "clean_power_count": int(details.get("clean_power_count", len(clean_values))),
            "median_power": float(details["baseline"]),
            "baseline_power": float(details["baseline"]),
            "aggregation": str(details.get("aggregation", baseline_details.get("aggregation", "median_after_iqr_filter"))),
        }
    return result


def _power_matrix_for_profile(baseline_details: dict[str, object]) -> dict[str, object]:
    return {
        "trial_ids": [str(trial_id) for trial_id in baseline_details.get("trial_ids", ())],
        "channels": [str(channel) for channel in baseline_details.get("channels", ACTION_CHANNELS)],
        "rows": [
            {str(channel): float(value) for channel, value in dict(row).items()}
            for row in baseline_details.get("power_rows", [])
        ],
        "values": [
            [
                float(dict(row)[channel])
                for channel in tuple(str(channel) for channel in baseline_details.get("channels", ACTION_CHANNELS))
            ]
            for row in baseline_details.get("power_rows", [])
        ],
    }


def _criterion_thresholds(
    *,
    subject_id: str,
    channel_baselines: dict[str, float],
    calibration_power_details: dict[str, object],
) -> tuple[float, dict[str, float], dict[str, object]]:
    action_threshold = float(
        get_mi_success_drop_threshold_pct(
            subject_id=subject_id,
            excluded_channel=None,
            used_channels=ACTION_CHANNELS,
            channel_baselines=channel_baselines,
            calibration_power_details=calibration_power_details,
        )
    )
    thresholds_without_channel: dict[str, float] = {}
    for excluded_channel in ACTION_CHANNELS:
        used_channels = tuple(channel for channel in ACTION_CHANNELS if channel != excluded_channel)
        thresholds_without_channel[excluded_channel] = float(
            get_mi_success_drop_threshold_pct(
                subject_id=subject_id,
                excluded_channel=excluded_channel,
                used_channels=used_channels,
                channel_baselines={
                    channel: float(channel_baselines[channel])
                    for channel in used_channels
                },
                calibration_power_details=calibration_power_details,
            )
        )
    criterion_details = {
        "method": "criterion_function",
        "criterion_function": "get_mi_success_drop_threshold_pct",
        "criterion_name": MI_SUCCESS_CRITERION_NAME,
        "criterion_version": MI_SUCCESS_CRITERION_VERSION,
        "threshold_pct": float(action_threshold),
        "T_all": float(action_threshold),
        "single_channel_thresholds": dict(thresholds_without_channel),
        "note": "20% is the current fixed protocol MI ROI Power Drop criterion; it is not learned from Rest calibration.",
    }
    return action_threshold, thresholds_without_channel, criterion_details


def _profile_extra(
    *,
    count: int,
    baseline_details: dict[str, object],
    criterion_details: dict[str, object],
) -> dict[str, object]:
    calibration_power_details = _power_details_for_profile(baseline_details)
    return {
        "required_rest_trials": int(count),
        "calibration_power_details": calibration_power_details,
        "calibration_power_matrix": _power_matrix_for_profile(baseline_details),
        "baseline_aggregation": "median_after_iqr_filter",
        "iqr_filter": {
            "method": "tukey_1.5_iqr",
            "multiplier": 1.5,
            "scope": "per_channel_once_after_signal_quality_accepted_rest_trials",
        },
        "power_definition": "mean_square_after_preprocessing",
        "threshold_method": "criterion_function",
        "criterion_details": dict(criterion_details),
        "roi_weights": {
            "left": {"C3": 1.0, "CP3": 0.9, "FC3": 0.8},
            "right": {"C4": 1.0, "CP4": 0.9, "FC4": 0.8},
        },
    }


def rest_loo_action_scores_for_channels(
    rest_trials: Sequence[TrialInput],
    *,
    required_rest_trials: int = DEFAULT_REQUIRED_REST_TRIALS,
    channels: Sequence[str] = ACTION_CHANNELS,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Deprecated compatibility shell: Rest-N calibration no longer computes LOO scores."""
    _ = rest_trials, required_rest_trials, channels
    return np.asarray([], dtype=np.float64), []


def rest_loo_action_score_details_for_channels(
    rest_trials: Sequence[TrialInput],
    *,
    required_rest_trials: int = DEFAULT_REQUIRED_REST_TRIALS,
    channels: Sequence[str] = ACTION_CHANNELS,
) -> tuple[np.ndarray, list[dict[str, float]], list[dict[str, object]]]:
    """Deprecated compatibility shell: Rest-N calibration no longer computes LOO details."""
    _ = rest_trials, required_rest_trials, channels
    return np.asarray([], dtype=np.float64), [], []


def rest_loo_action_scores(
    rest_trials: Sequence[TrialInput],
    *,
    required_rest_trials: int = DEFAULT_REQUIRED_REST_TRIALS,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Deprecated compatibility shell: Rest-N calibration no longer computes LOO scores."""
    return rest_loo_action_scores_for_channels(rest_trials, required_rest_trials=required_rest_trials)


def rest_loo_action_score_details(
    rest_trials: Sequence[TrialInput],
    *,
    required_rest_trials: int = DEFAULT_REQUIRED_REST_TRIALS,
) -> tuple[np.ndarray, list[dict[str, float]], list[dict[str, object]]]:
    """Deprecated compatibility shell: Rest-N calibration no longer computes LOO details."""
    return rest_loo_action_score_details_for_channels(rest_trials, required_rest_trials=required_rest_trials)


def rest10_loo_action_scores(rest_trials: Sequence[TrialInput]) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Deprecated compatibility wrapper for old Rest-10 LOO API."""
    return rest_loo_action_scores(rest_trials, required_rest_trials=DEFAULT_REQUIRED_REST_TRIALS)


def calibrate_rest(
    *,
    subject_id: str,
    rest_trials: Sequence[TrialInput],
    channel_names: Sequence[str],
    sampling_rate: float,
    required_rest_trials: int = DEFAULT_REQUIRED_REST_TRIALS,
) -> PersonalProfile:
    """Calibrate Personal Rest Power baselines from exactly N labeled rest trials."""
    count = validate_required_rest_trials(required_rest_trials)
    rest = tuple(rest_trials)
    _check_count(rest, count, "rest_trials")
    _check_labels(rest, REST_LABEL, "rest_trials")
    _check_sampling_and_channels(rest, channel_names, sampling_rate)

    baseline_details = build_rest_baseline_details(rest, channels=ACTION_CHANNELS)
    baselines = dict(baseline_details["baselines"])
    calibration_power_details = _power_details_for_profile(baseline_details)
    action_threshold, thresholds_without_channel, criterion_details = _criterion_thresholds(
        subject_id=subject_id,
        channel_baselines=baselines,
        calibration_power_details=calibration_power_details,
    )
    eeg_trace.trace_calibration_baseline_derivation(
        rest_power_details=baseline_details,
        criterion_details=criterion_details,
        thresholds_without_channel=thresholds_without_channel,
    )
    return _profile(
        subject_id=subject_id,
        mode=f"rest{count}",
        sampling_rate=sampling_rate,
        channel_names=channel_names,
        channel_baselines=baselines,
        action_threshold=action_threshold,
        action_thresholds_without_channel=thresholds_without_channel,
        laterality_threshold=REST10_LATERALITY_THRESHOLD,
        threshold_source_action="mi_success_drop_pct",
        threshold_source_laterality="deprecated_fixed_zero",
        calibration_trial_ids=[trial.trial_id for trial in rest],
        extra=_profile_extra(
            count=count,
            baseline_details=baseline_details,
            criterion_details=criterion_details,
        ),
    )


def calibrate_rest10(
    *,
    subject_id: str,
    rest_trials: Sequence[TrialInput],
    channel_names: Sequence[str],
    sampling_rate: float,
) -> PersonalProfile:
    """Compatibility wrapper for fixed ten-trial Rest-10 calibration."""
    return calibrate_rest(
        subject_id=subject_id,
        rest_trials=rest_trials,
        channel_names=channel_names,
        sampling_rate=sampling_rate,
        required_rest_trials=DEFAULT_REQUIRED_REST_TRIALS,
    )


def calibrate_cal10(
    *,
    subject_id: str,
    rest_trials: Sequence[TrialInput],
    left_trials: Sequence[TrialInput],
    right_trials: Sequence[TrialInput],
    channel_names: Sequence[str],
    sampling_rate: float,
) -> PersonalProfile:
    """Compatibility Cal-10 wrapper; formal thresholds are no longer learned from MI trials."""
    rest, left, right = tuple(rest_trials), tuple(left_trials), tuple(right_trials)
    _check_count(rest, 10, "rest_trials")
    _check_count(left, 10, "left_trials")
    _check_count(right, 10, "right_trials")
    _check_labels(rest, REST_LABEL, "rest_trials")
    _check_labels(left, LEFT_LABEL, "left_trials")
    _check_labels(right, RIGHT_LABEL, "right_trials")
    _check_sampling_and_channels(rest + left + right, channel_names, sampling_rate)
    return calibrate_rest(
        subject_id=subject_id,
        rest_trials=rest,
        channel_names=channel_names,
        sampling_rate=sampling_rate,
        required_rest_trials=10,
    )


def _baseline_details_from_power_array(
    *,
    rest: np.ndarray,
    channels: Sequence[str],
    calibration_trial_ids: Sequence[str],
) -> dict[str, object]:
    selected_channels = tuple(str(channel) for channel in channels)
    rows = [
        {channel: float(value) for channel, value in zip(selected_channels, row)}
        for row in np.asarray(rest, dtype=np.float64)
    ]
    baselines: dict[str, float] = {}
    channel_details: dict[str, dict[str, object]] = {}
    for index, channel in enumerate(selected_channels):
        values = np.asarray(rest[:, index], dtype=np.float64)
        filter_details = filter_rest_powers_by_iqr(
            trial_ids=tuple(str(trial_id) for trial_id in calibration_trial_ids),
            powers=values,
        )
        clean_values = np.asarray(filter_details["clean_powers"], dtype=np.float64)
        baseline = float(np.median(clean_values))
        if not np.isfinite(baseline) or baseline <= 0:
            raise InvalidCalibrationLabelError(f"Baseline for {channel} must be positive and finite.")
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
            "baseline": baseline,
            "aggregation": "median_after_iqr_filter",
        }
    return {
        "baselines": baselines,
        "power_rows": rows,
        "trial_ids": tuple(str(trial_id) for trial_id in calibration_trial_ids),
        "channels": selected_channels,
        "aggregation": "median_after_iqr_filter",
        "iqr_filter": {"method": "tukey_1.5_iqr", "multiplier": 1.5},
        "channel_details": channel_details,
    }


def calibrate_rest_from_powers(
    *,
    subject_id: str,
    rest_power: np.ndarray,
    channel_names: Sequence[str],
    sampling_rate: float,
    calibration_trial_ids: Sequence[str],
    required_rest_trials: int = DEFAULT_REQUIRED_REST_TRIALS,
) -> tuple[PersonalProfile, np.ndarray, list[dict[str, float]]]:
    """Calibrate Rest-N from precomputed [N, 6] channel powers."""
    count = validate_required_rest_trials(required_rest_trials)
    rest = np.asarray(rest_power, dtype=np.float64)
    if rest.shape != (count, len(ACTION_CHANNELS)):
        raise InvalidCalibrationCountError(f"Rest powers must be shaped [{count}, {len(ACTION_CHANNELS)}].")
    if not np.isfinite(rest).all():
        raise InvalidCalibrationLabelError("Rest powers must be finite.")
    if len(tuple(calibration_trial_ids)) != count:
        raise InvalidCalibrationCountError(f"calibration_trial_ids must contain exactly {count} trial IDs.")

    baseline_details = _baseline_details_from_power_array(
        rest=rest,
        channels=ACTION_CHANNELS,
        calibration_trial_ids=calibration_trial_ids,
    )
    baselines = dict(baseline_details["baselines"])
    calibration_power_details = _power_details_for_profile(baseline_details)
    action_threshold, thresholds_without_channel, criterion_details = _criterion_thresholds(
        subject_id=subject_id,
        channel_baselines=baselines,
        calibration_power_details=calibration_power_details,
    )
    profile = _profile(
        subject_id=subject_id,
        mode=f"rest{count}",
        sampling_rate=sampling_rate,
        channel_names=channel_names,
        channel_baselines=baselines,
        action_threshold=action_threshold,
        laterality_threshold=REST10_LATERALITY_THRESHOLD,
        action_thresholds_without_channel=thresholds_without_channel,
        threshold_source_action="mi_success_drop_pct",
        threshold_source_laterality="deprecated_fixed_zero",
        calibration_trial_ids=tuple(calibration_trial_ids),
        extra=_profile_extra(
            count=count,
            baseline_details=baseline_details,
            criterion_details=criterion_details,
        ),
    )
    return profile, np.asarray([], dtype=np.float64), []


def calibrate_rest10_from_powers(
    *,
    subject_id: str,
    rest_power: np.ndarray,
    channel_names: Sequence[str],
    sampling_rate: float,
    calibration_trial_ids: Sequence[str],
) -> tuple[PersonalProfile, np.ndarray, list[dict[str, float]]]:
    """Compatibility wrapper for fixed ten-trial Rest-10 power calibration."""
    return calibrate_rest_from_powers(
        subject_id=subject_id,
        rest_power=rest_power,
        channel_names=channel_names,
        sampling_rate=sampling_rate,
        calibration_trial_ids=calibration_trial_ids,
        required_rest_trials=DEFAULT_REQUIRED_REST_TRIALS,
    )


def calibrate_cal10_from_powers(
    *,
    subject_id: str,
    rest_power: np.ndarray,
    left_power: np.ndarray,
    right_power: np.ndarray,
    channel_names: Sequence[str],
    sampling_rate: float,
    calibration_trial_ids: Sequence[str],
) -> PersonalProfile:
    """Compatibility helper; formal thresholds are no longer learned from MI powers."""
    rest, left, right = (np.asarray(values, dtype=np.float64) for values in (rest_power, left_power, right_power))
    if rest.shape != (10, len(ACTION_CHANNELS)) or left.shape != rest.shape or right.shape != rest.shape:
        raise InvalidCalibrationCountError("Cal-10 powers must be shaped [10, 6] for rest, left, and right.")
    profile, _, _ = calibrate_rest_from_powers(
        subject_id=subject_id,
        rest_power=rest,
        channel_names=channel_names,
        sampling_rate=sampling_rate,
        calibration_trial_ids=tuple(calibration_trial_ids)[:10],
        required_rest_trials=10,
    )
    return profile
