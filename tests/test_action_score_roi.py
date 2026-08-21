from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from algo.erd import predict as predict_module
from algo.erd.algorithm_core import calibration as calibration_module
from algo.erd.algorithm_core import features as features_module
from algo.erd.algorithm_core.constants import ACTION_CHANNELS, PROFILE_SCHEMA_VERSION, REQUIRED_CHANNELS
from algo.erd.algorithm_core.criterion import get_mi_success_drop_threshold_pct
from algo.erd.algorithm_core.exceptions import InvalidProfileError
from algo.erd.algorithm_core.features import (
    compute_action_score_details,
    compute_channel_erd,
    compute_trial_channel_powers,
)
from algo.erd.algorithm_core.inference import _validate_profile, predict_trial
from algo.erd.algorithm_core.models import PersonalProfile, TrialInput
from algo.erd.algorithm_core.profile_io import load_profile
from algo.erd.algorithm_core.reporting import format_calibration_final_summary_report
from algo.erd.train import reset_session


TEST_TEMP_ROOT = Path(__file__).resolve().parent / "_tmp_power_drop_profiles"


def _profile(*, threshold: float = 20.0, schema_version: str = PROFILE_SCHEMA_VERSION) -> PersonalProfile:
    return PersonalProfile(
        profile_id="subject_rest10",
        subject_id="subject",
        mode="rest10",
        sampling_rate=200.0,
        channel_names=tuple(REQUIRED_CHANNELS),
        channel_baselines={channel: 100.0 for channel in ACTION_CHANNELS},
        action_threshold=threshold,
        laterality_threshold=0.0,
        threshold_source_action="mi_success_drop_pct",
        threshold_source_laterality="deprecated_fixed_zero",
        action_thresholds_without_channel={channel: threshold for channel in ACTION_CHANNELS},
        profile_schema_version=schema_version,
        calibration_trial_ids=tuple(f"R{index:02d}" for index in range(1, 11)),
        extra={"required_rest_trials": 10},
    )


def _trial() -> TrialInput:
    return TrialInput(
        trial_id="PRED_001",
        data=np.zeros((len(REQUIRED_CHANNELS), 800), dtype=np.float64),
        channel_names=tuple(REQUIRED_CHANNELS),
        sampling_rate=200.0,
        label=None,
    )


def _write_profile_json(profile_dir: Path, case_no: str, profile: PersonalProfile | None = None) -> Path:
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile = profile or _profile()
    payload = dataclasses.asdict(profile)
    payload["profile_id"] = f"{case_no}_rest10"
    payload["subject_id"] = case_no
    path = profile_dir / f"{case_no}_rest10_profile.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _remove_profile_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
        path.parent.rmdir()
    except OSError:
        pass


def test_mean_square_power_uses_preprocessed_trial_without_second_bandpass():
    data = np.zeros((len(REQUIRED_CHANNELS), 800), dtype=np.float64)
    c3_index = REQUIRED_CHANNELS.index("C3")
    data[c3_index, 100:500] = 2.0
    trial = TrialInput(
        trial_id="power",
        data=data,
        channel_names=tuple(REQUIRED_CHANNELS),
        sampling_rate=200.0,
        label="rest",
    )

    powers = compute_trial_channel_powers(trial, channels=("C3",))

    assert math.isclose(powers["C3"], 4.0)


def test_criterion_function_is_the_only_default_mi_threshold_source():
    assert math.isclose(get_mi_success_drop_threshold_pct(), 20.0)
    assert math.isclose(get_mi_success_drop_threshold_pct(excluded_channel="C3"), 20.0)
    assert math.isclose(
        get_mi_success_drop_threshold_pct(
            excluded_channel="FC4",
            used_channels=tuple(channel for channel in ACTION_CHANNELS if channel != "FC4"),
            channel_baselines={channel: 100.0 for channel in ACTION_CHANNELS},
            calibration_power_details={},
        ),
        20.0,
    )


def test_calibration_iqr_filters_outliers_per_channel_without_rejecting_trials():
    rest_power = np.asarray(
        [
            [30.0, 40.0, 50.0, 60.0, 20.0, 70.0],
            [31.0, 41.0, 51.0, 61.0, 21.0, 71.0],
            [32.0, 42.0, 52.0, 62.0, 22.0, 72.0],
            [33.0, 43.0, 53.0, 63.0, 23.0, 73.0],
            [34.0, 44.0, 54.0, 64.0, 138.0, 74.0],
            [35.0, 45.0, 55.0, 65.0, 24.0, 75.0],
            [36.0, 46.0, 56.0, 66.0, 25.0, 76.0],
            [37.0, 47.0, 57.0, 67.0, 26.0, 77.0],
            [38.0, 48.0, 58.0, 68.0, 27.0, 78.0],
            [39.0, 49.0, 59.0, 69.0, 28.0, 79.0],
        ],
        dtype=np.float64,
    )
    trial_ids = tuple(f"R{index:02d}" for index in range(1, 11))

    profile, deprecated_scores, deprecated_baselines = calibration_module.calibrate_rest_from_powers(
        subject_id="iqr",
        rest_power=rest_power,
        channel_names=REQUIRED_CHANNELS,
        sampling_rate=200.0,
        calibration_trial_ids=trial_ids,
        required_rest_trials=10,
    )

    assert deprecated_scores.size == 0
    assert deprecated_baselines == []
    assert profile.extra["baseline_aggregation"] == "median_after_iqr_filter"
    assert profile.extra["iqr_filter"]["method"] == "tukey_1.5_iqr"
    assert profile.calibration_trial_ids == trial_ids

    details = profile.extra["calibration_power_details"]
    cp3 = details["CP3"]
    cp3_raw = rest_power[:, ACTION_CHANNELS.index("CP3")]
    q1 = float(np.percentile(cp3_raw, 25))
    q3 = float(np.percentile(cp3_raw, 75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    expected_clean = [float(value) for value in cp3_raw if lower <= value <= upper]

    assert cp3["trial_powers"] == [float(value) for value in cp3_raw]
    assert cp3["sorted_powers"] == [float(value) for value in sorted(cp3_raw)]
    assert cp3["clean_powers"] == expected_clean
    assert cp3["clean_power_count"] == len(expected_clean)
    assert cp3["aggregation"] == "median_after_iqr_filter"
    assert math.isclose(cp3["iqr_filter"]["q1"], q1)
    assert math.isclose(cp3["iqr_filter"]["q3"], q3)
    assert math.isclose(cp3["iqr_filter"]["iqr"], iqr)
    assert math.isclose(cp3["iqr_filter"]["lower_fence"], lower)
    assert math.isclose(cp3["iqr_filter"]["upper_fence"], upper)
    assert cp3["iqr_filter"]["outliers"] == [
        {"trial_id": "R05", "power": 138.0, "reason": "above_upper_fence"}
    ]
    assert math.isclose(profile.channel_baselines["CP3"], float(np.median(expected_clean)))

    c3 = details["C3"]
    c3_raw = rest_power[:, ACTION_CHANNELS.index("C3")]
    assert c3["trial_powers"][4] == 54.0
    assert c3["clean_powers"] == [float(value) for value in c3_raw]
    assert c3["iqr_filter"]["outliers"] == []
    assert math.isclose(profile.channel_baselines["C3"], float(np.median(c3_raw)))


def test_calibration_final_report_explains_iqr_cleaned_baseline():
    rest_power = np.asarray(
        [
            [30.0, 40.0, 50.0, 60.0, 20.0, 70.0],
            [31.0, 41.0, 51.0, 61.0, 21.0, 71.0],
            [32.0, 42.0, 52.0, 62.0, 22.0, 72.0],
            [33.0, 43.0, 53.0, 63.0, 23.0, 73.0],
            [34.0, 44.0, 54.0, 64.0, 138.0, 74.0],
            [35.0, 45.0, 55.0, 65.0, 24.0, 75.0],
            [36.0, 46.0, 56.0, 66.0, 25.0, 76.0],
            [37.0, 47.0, 57.0, 67.0, 26.0, 77.0],
            [38.0, 48.0, 58.0, 68.0, 27.0, 78.0],
            [39.0, 49.0, 59.0, 69.0, 28.0, 79.0],
        ],
        dtype=np.float64,
    )
    profile, _, _ = calibration_module.calibrate_rest_from_powers(
        subject_id="iqr_report",
        rest_power=rest_power,
        channel_names=REQUIRED_CHANNELS,
        sampling_rate=200.0,
        calibration_trial_ids=tuple(f"R{index:02d}" for index in range(1, 11)),
        required_rest_trials=10,
    )

    text = format_calibration_final_summary_report(dataclasses.asdict(profile))

    assert "IQR" in text
    assert "Q1" in text
    assert "Q3" in text
    assert "Lower Fence" in text
    assert "Upper Fence" in text
    assert "Clean Rest Powers" in text
    assert "R05=138.000000 above_upper_fence" in text
    assert "Baseline      : median(Clean Rest Powers) = 24.000000" in text


def test_calibration_from_power_builds_independent_channel_median_baselines_and_no_rest_erd():
    rest_power = np.asarray(
        [
            [98.0, 110.0, 50.0, 210.0, 300.0, 410.0],
            [105.0, 111.0, 53.0, 211.0, 301.0, 411.0],
            [100.0, 112.0, 52.0, 212.0, 302.0, 412.0],
            [102.0, 113.0, 51.0, 213.0, 303.0, 413.0],
            [95.0, 114.0, 54.0, 214.0, 304.0, 414.0],
            [101.0, 115.0, 55.0, 215.0, 305.0, 415.0],
        ],
        dtype=np.float64,
    )

    profile, deprecated_scores, deprecated_baselines = calibration_module.calibrate_rest_from_powers(
        subject_id="median",
        rest_power=rest_power,
        channel_names=REQUIRED_CHANNELS,
        sampling_rate=200.0,
        calibration_trial_ids=tuple(f"R{index:02d}" for index in range(1, 7)),
        required_rest_trials=6,
    )

    assert deprecated_scores.size == 0
    assert deprecated_baselines == []
    assert profile.profile_schema_version == PROFILE_SCHEMA_VERSION
    assert profile.threshold_source_action == "mi_success_drop_pct"
    assert profile.extra["threshold_method"] == "criterion_function"
    assert profile.extra["baseline_aggregation"] == "median_after_iqr_filter"
    assert "loo_rest_action_scores" not in profile.extra
    assert "loo_action_summaries" not in profile.extra
    assert "rest_threshold_stats" not in profile.extra
    assert "robust_sigma_multiplier" not in profile.extra
    np.testing.assert_allclose(
        [profile.channel_baselines[channel] for channel in ACTION_CHANNELS],
        np.median(rest_power, axis=0),
    )
    assert profile.channel_baselines["FC3"] == float(np.median(rest_power[:, 0]))
    assert profile.channel_baselines["C3"] == float(np.median(rest_power[:, 2]))
    details = profile.extra["calibration_power_details"]
    assert details["FC3"]["trial_powers"] == [float(value) for value in rest_power[:, 0]]
    assert details["FC3"]["sorted_powers"] == [float(value) for value in sorted(rest_power[:, 0])]
    assert details["FC3"]["clean_powers"] == [float(value) for value in rest_power[:, 0]]
    assert details["FC3"]["clean_power_count"] == int(rest_power.shape[0])
    assert details["FC3"]["median_power"] == float(np.median(rest_power[:, 0]))
    assert details["FC3"]["aggregation"] == "median_after_iqr_filter"
    for channel in ACTION_CHANNELS:
        assert math.isclose(profile.action_thresholds_without_channel[channel], 20.0)
    assert math.isclose(profile.action_threshold, 20.0)


def test_calibrate_rest_does_not_call_rest_erd_or_action_score(monkeypatch):
    rows = {
        f"R{index:02d}": {channel: 10.0 * (offset + 1) + index for offset, channel in enumerate(ACTION_CHANNELS)}
        for index in range(1, 11)
    }

    def fake_powers(trial, *, channels=ACTION_CHANNELS):
        source = rows[str(trial.trial_id)]
        return {channel: float(source[channel]) for channel in channels}

    def fail_erd(*args, **kwargs):
        raise AssertionError("calibration must not compute rest ERD")

    def fail_score(*args, **kwargs):
        raise AssertionError("calibration must not compute rest action score")

    monkeypatch.setattr(features_module, "compute_trial_channel_powers", fake_powers)
    monkeypatch.setattr(calibration_module, "compute_channel_erd", fail_erd, raising=False)
    monkeypatch.setattr(calibration_module, "compute_action_score_details", fail_score, raising=False)

    trials = [
        TrialInput(
            trial_id=f"R{index:02d}",
            data=np.zeros((len(REQUIRED_CHANNELS), 800), dtype=np.float64),
            channel_names=tuple(REQUIRED_CHANNELS),
            sampling_rate=200.0,
            label="rest",
        )
        for index in range(1, 11)
    ]

    profile = calibration_module.calibrate_rest(
        subject_id="no_rest_erd",
        rest_trials=trials,
        channel_names=REQUIRED_CHANNELS,
        sampling_rate=200.0,
        required_rest_trials=10,
    )

    assert math.isclose(profile.action_threshold, 20.0)
    assert "calibration_power_details" in profile.extra
    assert "loo_rest_action_scores" not in profile.extra


def test_prediction_channel_erd_drop_selected_roi_and_threshold_boundary(monkeypatch):
    def fake_powers_action(trial, *, channels=ACTION_CHANNELS):
        return {channel: 80.0 for channel in channels}

    monkeypatch.setattr(features_module, "compute_trial_channel_powers", fake_powers_action)
    result = predict_trial(trial=_trial(), profile=_profile(threshold=20.0))

    assert result.stage1_result == "action"
    assert result.final_result == "action"
    assert result.stage2_result is None
    assert math.isclose(result.action_score, 20.0)
    assert math.isclose(result.action_threshold, 20.0)
    assert result.action_score_details["selected_roi"] == "left"
    assert math.isclose(result.action_score_details["left"]["erd_pct"], -20.0)
    assert math.isclose(result.action_score_details["left"]["drop_pct"], 20.0)

    def fake_powers_rest(trial, *, channels=ACTION_CHANNELS):
        return {channel: 80.001 for channel in channels}

    monkeypatch.setattr(features_module, "compute_trial_channel_powers", fake_powers_rest)
    result = predict_trial(trial=_trial(), profile=_profile(threshold=20.0))

    assert result.stage1_result == "rest"
    assert result.final_result == "rest"
    assert result.stage2_result is None
    assert result.action_score < result.action_threshold


def test_old_profile_schema_is_rejected():
    profile = _profile(schema_version="1.0.0")

    with pytest.raises(InvalidProfileError, match="requires recalibration"):
        _validate_profile(profile)


def test_pre_iqr_profile_schema_is_rejected():
    profile = _profile(schema_version="2.0.0")

    with pytest.raises(InvalidProfileError, match="requires recalibration"):
        _validate_profile(profile)


def test_profile_io_rejects_old_schema_version():
    profile_dir = TEST_TEMP_ROOT / "old_schema"
    path = _write_profile_json(profile_dir, "old_schema", _profile(schema_version="1.0.0"))

    try:
        with pytest.raises(InvalidProfileError, match="requires recalibration"):
            load_profile(path)
    finally:
        _remove_profile_file(path)


def _install_prediction_entry_mocks(monkeypatch, *, bad_channels: tuple[str, ...]) -> None:
    def fake_required_channels(filtered_samples, standard_mapping):
        return np.zeros((len(REQUIRED_CHANNELS), 800), dtype=np.float64)

    def fake_trial_input(**kwargs):
        return TrialInput(
            trial_id=f"{kwargs['case_no']}_predict_001",
            data=np.zeros((len(REQUIRED_CHANNELS), 800), dtype=np.float64),
            channel_names=tuple(REQUIRED_CHANNELS),
            sampling_rate=float(kwargs["sample_rate"]),
            label=None,
        )

    def fake_quality(trial):
        return {
            "passed": len(bad_channels) == 0,
            "evaluation_succeeded": True,
            "channels": [
                {
                    "name": channel,
                    "status": "poor" if channel in bad_channels else "good",
                }
                for channel in ACTION_CHANNELS
            ],
            "bad_channels": list(bad_channels),
            "failed_channels": list(bad_channels),
        }

    monkeypatch.setattr(predict_module, "backend_samples_to_required_channels", fake_required_channels)
    monkeypatch.setattr(predict_module, "make_preprocessed_trial_input", fake_trial_input)
    monkeypatch.setattr(predict_module, "evaluate_prediction_window_quality", fake_quality)


def test_single_bad_c3_prediction_uses_t_no_c3_dynamic_denominator_and_decision_detail(monkeypatch):
    case_no = "single_bad_c3"
    profile_dir = TEST_TEMP_ROOT / "single_bad_c3"
    profile_path = _write_profile_json(profile_dir, case_no)
    reset_session(case_no)
    _install_prediction_entry_mocks(monkeypatch, bad_channels=("C3",))

    def fake_channel_powers(trial, *, channels=ACTION_CHANNELS):
        assert tuple(channels) == tuple(channel for channel in ACTION_CHANNELS if channel != "C3")
        return {channel: 80.0 for channel in channels}

    monkeypatch.setattr(predict_module, "compute_trial_channel_powers", fake_channel_powers)

    try:
        result = predict_module.predict_rest10_stream(
            case_no=case_no,
            sample_rate=200,
            filtered_samples=np.zeros((800, 1), dtype=np.float64),
            standard_mapping={},
            profile_dir=profile_dir,
        )
    finally:
        reset_session(case_no)
        _remove_profile_file(profile_path)

    assert result["success"] is True
    assert result["code"] == 1
    assert result["quality_mode"] == "single_channel_excluded"
    assert result["threshold_source"] == "T_no_C3"
    assert math.isclose(float(result["action_threshold"]), 20.0)
    assert math.isclose(float(result["action_score"]), 20.0)
    assert "C3" not in result["channel_erd"]
    assert math.isclose(float(result["report_summary"]["left_weight_sum"]), 1.7)
    assert math.isclose(float(result["report_summary"]["right_weight_sum"]), 2.7)
    assert result["decision_detail"]["excluded_channel"] == "C3"
    assert result["decision_detail"]["criterion_met"] is True


def test_two_bad_action_channels_still_invalid(monkeypatch):
    case_no = "two_bad"
    profile_dir = TEST_TEMP_ROOT / "two_bad"
    profile_path = _write_profile_json(profile_dir, case_no)
    reset_session(case_no)
    _install_prediction_entry_mocks(monkeypatch, bad_channels=("C3", "FC4"))

    try:
        result = predict_module.predict_rest10_stream(
            case_no=case_no,
            sample_rate=200,
            filtered_samples=np.zeros((800, 1), dtype=np.float64),
            standard_mapping={},
            profile_dir=profile_dir,
        )
    finally:
        reset_session(case_no)
        _remove_profile_file(profile_path)

    assert result["success"] is True
    assert result["code"] == -1
    assert result["classification_available"] is False
    assert result["action_score"] is None
    assert result["action_threshold"] is None


def test_action_score_details_reports_erd_and_drop_aliases():
    erd = {
        "C3": -30.0,
        "CP3": -20.0,
        "FC3": -10.0,
        "C4": -5.0,
        "CP4": -4.0,
        "FC4": 2.0,
    }

    details = compute_action_score_details(erd)

    assert math.isclose(float(details["left"]["erd_pct"]), -20.74074074074074)
    assert math.isclose(float(details["left"]["drop_pct"]), 20.74074074074074)
    assert math.isclose(float(details["action_score"]), float(details["selected_roi_drop_pct"]))
    assert details["selected_roi"] == "left"


def test_channel_erd_defines_prediction_power_drop():
    erd = compute_channel_erd({"C3": 80.0}, {"C3": 100.0})

    assert math.isclose(erd["C3"], -20.0)
    assert math.isclose(-erd["C3"], 20.0)
