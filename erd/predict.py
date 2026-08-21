"""Streaming Rest-N prediction adapter."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from pathlib import Path
import logging
import time

import numpy as np

from algo.erd.algorithm_core.calibration import (validate_required_rest_trials,)
from algo.erd.algorithm_core.constants import (
    ACTION_CHANNELS,
    ACTION_LABEL,
    DEFAULT_REQUIRED_REST_TRIALS,
    PREDICTION_ACTION,
    PREDICTION_INVALID_QUALITY,
    PREDICTION_REST,
    REQUIRED_CHANNELS,
    REST_LABEL,
)
from algo.erd.algorithm_core.exceptions import (InvalidRequiredRestTrialsError,)
from algo.erd.algorithm_core.features import (
    compute_action_score,
    compute_action_score_details,
    compute_channel_erd,
    compute_trial_channel_powers,
)
from algo.erd.algorithm_core.inference import (predict_trial,)
from algo.erd.algorithm_core.profile_io import (load_profile,)
from algo.erd.algorithm_core.reporting import build_prediction_report_summary
from algo.erd.algorithm_core import trace as eeg_trace
from algo.erd.train import (DEFAULT_PROFILE_DIR,PREDICTION_TYPE,TRIAL_SECONDS,append_stream_buffer,
    backend_samples_to_required_channels,
    evaluate_rest_trial_quality,
    get_session,
    make_preprocessed_trial_input,
    profile_path_for_case,
    validate_stream_sample_rate,
)


logger = logging.getLogger(__name__)
PREDICTION_QUALITY_CHANNELS: tuple[str, ...] = ACTION_CHANNELS
PREDICTION_ACCEPTED_QUALITY_STATUSES = frozenset({"good", "acceptable"})


def _normalize_backend_trial_count(
    backend_trial_count: object,
) -> int | None:
    if backend_trial_count is None:
        return None
    try:
        return int(backend_trial_count)
    except (TypeError, ValueError, OverflowError):
        return None


def _advance_trial_prediction_index(
    session,
    backend_trial_count: object,
) -> tuple[int | None, int | None]:
    normalized_count = _normalize_backend_trial_count(
        backend_trial_count
    )
    if normalized_count is None:
        return None, None

    if session.prediction_backend_trial_count != normalized_count:
        session.prediction_backend_trial_count = normalized_count
        session.trial_prediction_index = 1
    else:
        session.trial_prediction_index += 1

    return normalized_count, int(session.trial_prediction_index)


def _invalid_count_result(
    exc: Exception,
) -> dict[str, object]:
    """返回静息校准次数配置错误。"""

    return {
        "success": False,
        "code": -1,
        "status": "invalid_required_rest_trials",
        "message": str(exc),
    }


def validate_prediction_step_seconds(
    prediction_step_seconds: object,
    *,
    sample_rate: int | float,
    window_seconds: float = TRIAL_SECONDS,
) -> int:
    """校验滑动步长，并转换为采样点数。"""

    if (
        isinstance(
            prediction_step_seconds,
            (bool, np.bool_),
        )
        or not isinstance(
            prediction_step_seconds,
            Real,
        )
    ):
        raise ValueError(
            "prediction_step_seconds must be "
            "a positive number, not bool or string."
        )

    seconds = float(
        prediction_step_seconds
    )

    if (
        not np.isfinite(seconds)
        or seconds <= 0
        or seconds > float(window_seconds)
    ):
        raise ValueError(
            "prediction_step_seconds must be > 0 "
            f"and <= {float(window_seconds)}, "
            f"got {prediction_step_seconds!r}."
        )

    samples = int(
        round(
            seconds * float(sample_rate)
        )
    )

    if samples <= 0:
        raise ValueError(
            "prediction_step_seconds is too small "
            "for the current sample rate."
        )

    return samples


def _invalid_step_result(
    exc: Exception,
) -> dict[str, object]:
    """返回预测滑动步长配置错误。"""

    return {
        "success": False,
        "code": -1,
        "status": "invalid_prediction_step_seconds",
        "message": str(exc),
    }


def evaluate_prediction_window_quality(trial) -> dict[str, object]:
    """Evaluate the current preprocessed prediction window without touching live quality state."""
    try:
        quality = evaluate_rest_trial_quality(trial)
        if not isinstance(quality, Mapping):
            raise ValueError("quality result must be a dict-like object.")
        failed_channels = [str(channel) for channel in quality.get("failed_channels", [])]
        channels = quality.get("channels", [])
        if not isinstance(channels, (list, tuple)):
            raise ValueError("quality result channels must be a list.")
        status_by_name: dict[str, str] = {}
        for item in channels:
            if isinstance(item, Mapping):
                name = str(item.get("name", ""))
                if name:
                    status_by_name[name] = str(item.get("status", "unknown"))
        missing_status = [
            channel for channel in PREDICTION_QUALITY_CHANNELS
            if channel not in status_by_name
        ]
        if missing_status and "error" not in quality:
            failed_channels = sorted(set(failed_channels + missing_status), key=PREDICTION_QUALITY_CHANNELS.index)
        bad_channels = [
            channel for channel in PREDICTION_QUALITY_CHANNELS
            if channel in set(failed_channels)
            or status_by_name.get(channel, "unknown") not in PREDICTION_ACCEPTED_QUALITY_STATUSES
        ]
        evaluation_succeeded = "error" not in quality and not missing_status
        return {
            **dict(quality),
            "evaluation_succeeded": bool(evaluation_succeeded),
            "bad_channels": bad_channels,
            "failed_channels": failed_channels,
        }
    except Exception as exc:
        return {
            "passed": False,
            "evaluation_succeeded": False,
            "channels": [],
            "bad_channels": list(PREDICTION_QUALITY_CHANNELS),
            "failed_channels": list(PREDICTION_QUALITY_CHANNELS),
            "error": str(exc),
        }


def calculate_action_score(
    erd_values: dict[str, float],
    used_channels: tuple[str, ...],
) -> float:
    return compute_action_score(
        erd_values,
        channels=used_channels,
    )


def get_threshold_without_channel(
    profile,
    excluded_channel: str,
) -> float | None:
    thresholds = dict(
        getattr(
            profile,
            "action_thresholds_without_channel",
            {},
        )
        or {}
    )
    value = thresholds.get(str(excluded_channel))
    if value is None:
        return None
    threshold = float(value)
    if not np.isfinite(threshold):
        return None
    return threshold


def build_invalid_quality_result(
    *,
    case_no: str,
    profile,
    profile_source: str,
    calibration_bypassed: bool,
    reason: str,
    bad_channels=(),
    buffer_samples: int,
    required_samples: int,
    prediction_step_samples: int,
    prediction_window_ready_epoch: float,
    prediction_window_ready_perf: float,
    prediction_result_epoch: float,
    prediction_result_perf: float,

    # 当前预测窗口诊断信息
    prediction_index: int,
    profile_path: str | Path,
    window_start_sample: int,
    window_end_sample: int,
    window_start_s: float,
    window_end_s: float,
    prediction_compute_ms: float,

    quality_result: dict[str, object] | None = None,
    fallback_profile_path: str | Path | None = None,
) -> dict[str, object]:
    response: dict[str, object] = {
        "success": True,
        "code": PREDICTION_INVALID_QUALITY,
        "status": "predicted",
        "quality_status": "prediction_invalid_quality",
        "classification_available": False,
        "reason": str(reason),
        "bad_channels": [str(channel) for channel in bad_channels],
        "case_no": str(case_no),
        "profile_id": profile.profile_id,
        "profile_source": profile_source,
        "calibration_bypassed": bool(calibration_bypassed),
        "label": "invalid_quality",
        "stage1_result": "invalid_quality",
        "stage2_result": None,
        "final_result": "invalid_quality",
        "action_score": None,
        "action_threshold": None,
        "buffer_samples": int(buffer_samples),
        "required_samples": int(required_samples),
        "prediction_step_samples": int(prediction_step_samples),
        "prediction_window_ready_epoch": float(prediction_window_ready_epoch),
        "prediction_window_ready_perf": float(prediction_window_ready_perf),
        "prediction_result_epoch": float(prediction_result_epoch),
        "prediction_result_perf": float(prediction_result_perf),
        "message": ("prediction window invalid because channel quality is unreliable"),
        # 新增
        "profile_path": str(profile_path),
        "prediction_index": int(prediction_index),
        # 当前预测窗口位置
        "window_start_sample": int(window_start_sample),
        "window_end_sample": int(window_end_sample),
        "window_start_s": float(window_start_s),
        "window_end_s": float(window_end_s),
        "prediction_compute_ms": float(prediction_compute_ms),
    }

    if quality_result is not None:
        response["quality_result"] = dict(quality_result)
        if "error" in quality_result:
            response["quality_error"] = str(quality_result["error"])
    if fallback_profile_path is not None:
        response["fallback_profile_path"] = str(fallback_profile_path)
    return response


def predict_rest10_stream(
    *,
    case_no: str,
    sample_rate: int,
    filtered_samples: np.ndarray,
    standard_mapping: Mapping[str, int],
    profile_dir: str | Path = DEFAULT_PROFILE_DIR,
    type: int = PREDICTION_TYPE,
    required_rest_trials: int = DEFAULT_REQUIRED_REST_TRIALS,
    allow_uncalibrated_prediction: bool = False,
    fallback_profile_path: str | Path | None = None,
    prediction_step_seconds: float = 1.0,
    backend_trial_count: int | None = None,
) -> dict[str, object]:
    """缓存流式EEG，并使用四秒滑动窗口完成预测。

    第一次预测需要收满四秒数据。
    后续预测按照 prediction_step_seconds 对窗口进行前移。
    """

    try:
        count = validate_required_rest_trials(
            required_rest_trials
        )
    except InvalidRequiredRestTrialsError as exc:
        return _invalid_count_result(exc)

    try:
        rate = validate_stream_sample_rate(
            sample_rate
        )
    except ValueError as exc:
        return {
            "success": False,
            "code": -1,
            "status": "invalid_sample_rate",
            "message": str(exc),
        }

    try:
        step_samples = (
            validate_prediction_step_seconds(
                prediction_step_seconds,
                sample_rate=rate,
            )
        )
    except ValueError as exc:
        return _invalid_step_result(exc)

    if int(type) != PREDICTION_TYPE:
        return {
            "success": False,
            "code": -1,
            "status": "invalid_type",
            "message": (
                "predict_rest10_stream expects "
                "type=1 prediction."
            ),
        }

    personal_path = profile_path_for_case(
        case_no,
        profile_dir,
        required_rest_trials=count,
    )

    if personal_path.exists():
        # 优先使用当前用户的个人Profile。
        profile_path = personal_path
        profile_source = "personal"
        calibration_bypassed = False

    else:
        if not allow_uncalibrated_prediction:
            return {
                "success": False,
                "code": -1,
                "status": "no_profile",
                "case_no": str(case_no),
                "profile_path": str(personal_path),
                "message": (
                    "Personal calibration profile "
                    "not found; calibration is "
                    "required first."
                ),
            }

        if fallback_profile_path is None:
            return {
                "success": False,
                "code": -1,
                "status": (
                    "missing_fallback_profile_path"
                ),
                "message": (
                    "Uncalibrated prediction was "
                    "enabled, but "
                    "fallback_profile_path was not "
                    "provided."
                ),
            }

        profile_path = Path(
            fallback_profile_path
        )

        if not profile_path.exists():
            return {
                "success": False,
                "code": -1,
                "status": "no_fallback_profile",
                "message": (
                    "Uncalibrated prediction was "
                    "enabled, but no valid fallback "
                    "profile was available."
                ),
            }

        profile_source = "fallback"
        calibration_bypassed = True

    try:
        profile = load_profile(
            profile_path
        )
    except Exception as exc:
        if profile_source == "fallback":
            return {
                "success": False,
                "code": -1,
                "status": (
                    "invalid_fallback_profile"
                ),
                "message": str(exc),
                "fallback_profile_path": str(
                    profile_path
                ),
            }

        return {
            "success": False,
            "code": -1,
            "status": "invalid_personal_profile",
            "message": str(exc),
        }

    if not np.isclose(
        float(profile.sampling_rate),
        rate,
    ):
        return {
            "success": False,
            "code": -1,
            "status": (
                "profile_configuration_mismatch"
            ),
            "message": (
                "Profile sample_rate="
                f"{profile.sampling_rate:g} Hz "
                "does not match input sample_rate="
                f"{rate:g} Hz."
            ),
            "profile_path": str(profile_path),
        }

    session = get_session(
        case_no,
        required_rest_trials=count,
    )

    trial_samples = int(
        round(
            rate * TRIAL_SECONDS
        )
    )

    if session.prediction_window_samples is None:
        session.prediction_window_samples = (
            trial_samples
        )
        session.prediction_step_samples = (
            step_samples
        )
        session.prediction_sample_rate = rate

    elif (
        session.prediction_window_samples
        != trial_samples
        or session.prediction_step_samples
        != step_samples
        or session.prediction_sample_rate is None
        or not np.isclose(
            session.prediction_sample_rate,
            rate,
        )
    ):
        return {
            "success": False,
            "code": -1,
            "status": (
                "prediction_configuration_mismatch"
            ),
            "message": (
                "prediction window or step changed "
                "during an active prediction "
                "session; reset the session before "
                "changing configuration."
            ),
        }

    eeg_trace.trace_session(
        case_no=str(case_no),
        mode="prediction",
        sample_rate=rate,
        input_channels=int(np.asarray(filtered_samples).shape[1]) if np.asarray(filtered_samples).ndim == 2 else 0,
        required_channels=REQUIRED_CHANNELS,
        action_channels=PREDICTION_QUALITY_CHANNELS,
        window_seconds=TRIAL_SECONDS,
        step_seconds=prediction_step_seconds,
        mapping=standard_mapping,
    )

    required_channels = (
        backend_samples_to_required_channels(
            filtered_samples,
            standard_mapping,
        )
    )

    session.prediction_buffer = (
        append_stream_buffer(
            session.prediction_buffer,
            required_channels,
        )
    )

    current_samples = int(
        session.prediction_buffer.shape[1]
    )

    if current_samples < trial_samples:
        return {
            "success": True,
            "code": 0,
            "status": "collecting",
            "case_no": str(case_no),
            "buffer_samples": current_samples,
            "required_samples": trial_samples,
            "message": (
                "Collecting prediction samples."
            ),
        }

    # 使用最前面的四秒作为本次预测窗口。
    raw_trial = (
        session.prediction_buffer[
            :,
            :trial_samples,
        ].copy()
    )

    # 预测后按照滑动步长前移。
    session.prediction_buffer = (
        session.prediction_buffer[
            :,
            step_samples:,
        ].copy()
    )

    session.prediction_trial_count += 1

    prediction_index = int(
        session.prediction_trial_count
    )

    backend_trial_count_for_trace, trial_prediction_index = (
        _advance_trial_prediction_index(
            session,
            backend_trial_count,
        )
    )

    window_start_sample = int(
        (prediction_index - 1) * step_samples
    )

    window_end_sample = int(
        window_start_sample + trial_samples
    )

    window_start_s = float(
        window_start_sample / rate
    )

    window_end_s = float(
        window_end_sample / rate
    )

    # T2：当前4秒窗口已经收满，准备进入预处理和预测。
    prediction_window_ready_epoch = time.time()

    # 专门用于计算耗时，不受系统时间调整影响。
    prediction_window_ready_perf = time.perf_counter()

    try:
        trial = make_preprocessed_trial_input(
            case_no=str(case_no),
            role="predict",
            index=session.prediction_trial_count,
            raw_data_8x_samples=raw_trial,
            sample_rate=rate,
            label=None,
        )

    except Exception as exc:
        if profile_source == "fallback":
            return {
                "success": False,
                "code": -1,
                "status": "invalid_fallback_profile",
                "message": str(exc),
                "fallback_profile_path": str(profile_path),
            }

        return {
            "success": False,
            "code": -1,
            "status": "invalid_personal_profile",
            "message": str(exc),
        }

    def invalid_quality(reason: str, bad_channels, quality_result: dict[str, object] | None = None) -> dict[str, object]:
        prediction_result_epoch = time.time()
        prediction_result_perf = time.perf_counter()

        prediction_index = int(
            session.prediction_trial_count
        )

        window_start_sample = int(
            (prediction_index - 1) * step_samples
        )

        window_end_sample = int(
            window_start_sample + trial_samples
        )

        window_start_s = float(
            window_start_sample / rate
        )

        window_end_s = float(
            window_end_sample / rate
        )

        prediction_compute_ms = float(
            (
                    prediction_result_perf
                    - prediction_window_ready_perf
            )
            * 1000.0
        )

        logger.info(
            "prediction window invalid: code=-1, reason=%s, bad_channels=%s",
            reason,
            list(bad_channels),
        )
        eeg_trace.trace_prediction_window_summary(
            backend_trial_count=backend_trial_count_for_trace,
            trial_prediction_index=trial_prediction_index,
            prediction_index=prediction_index,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
            sample_rate=rate,
            channel_count=len(REQUIRED_CHANNELS),
            sample_count=trial_samples,
            quality_mode=f"invalid_quality:{reason}",
            used_channels=(),
            bad_channels=[str(channel) for channel in bad_channels],
            channel_powers={},
            channel_baselines={},
            channel_erd={},
            roi_details=None,
            threshold_source="invalid_quality",
            action_score=None,
            action_threshold=None,
            score_margin=None,
            preprocess_summary=getattr(trial, "preprocess_summary", {}),
            code=PREDICTION_INVALID_QUALITY,
        )
        return build_invalid_quality_result(
            case_no=str(case_no),
            profile=profile,
            profile_source=profile_source,
            calibration_bypassed=calibration_bypassed,
            reason=reason,
            bad_channels=bad_channels,
            buffer_samples=int(session.prediction_buffer.shape[1]),
            required_samples=trial_samples,
            prediction_step_samples=step_samples,
            prediction_window_ready_epoch=float(prediction_window_ready_epoch),
            prediction_window_ready_perf=float(prediction_window_ready_perf),
            prediction_result_epoch=float(prediction_result_epoch),
            prediction_result_perf=float(prediction_result_perf),
            #新增
            prediction_index=(prediction_index),
            profile_path=(profile_path),
            window_start_sample=(window_start_sample),
            window_end_sample=(window_end_sample),
            window_start_s=(window_start_s),
            window_end_s=(window_end_s),
            prediction_compute_ms=(prediction_compute_ms),
            quality_result=quality_result,
            fallback_profile_path=(profile_path if profile_source == "fallback" else None),
        )

    try:
        quality_result = evaluate_prediction_window_quality(
            trial
        )
    except Exception as exc:
        quality_result = {
            "passed": False,
            "evaluation_succeeded": False,
            "channels": [],
            "bad_channels": list(PREDICTION_QUALITY_CHANNELS),
            "failed_channels": list(PREDICTION_QUALITY_CHANNELS),
            "error": str(exc),
        }

    if not isinstance(quality_result, Mapping):
        quality_result = {
            "passed": False,
            "evaluation_succeeded": False,
            "channels": [],
            "bad_channels": list(PREDICTION_QUALITY_CHANNELS),
            "failed_channels": list(PREDICTION_QUALITY_CHANNELS),
            "error": "quality result must be a dict-like object.",
        }

    if not bool(quality_result.get("evaluation_succeeded")):
        return invalid_quality(
            "quality_evaluation_failed",
            quality_result.get("bad_channels", PREDICTION_QUALITY_CHANNELS),
            dict(quality_result),
        )

    bad_channels = [
        str(channel)
        for channel in quality_result.get("bad_channels", [])
        if str(channel) in PREDICTION_QUALITY_CHANNELS
    ]

    if "C3" in bad_channels and "C4" in bad_channels:
        return invalid_quality(
            "c3_c4_both_bad",
            bad_channels,
            dict(quality_result),
        )

    if len(bad_channels) >= 2:
        return invalid_quality(
            "multiple_bad_channels",
            bad_channels,
            dict(quality_result),
        )

    channel_powers: dict[str, float] = {}
    channel_baselines: dict[str, float] = {}
    erd_values: dict[str, float] = {}
    roi_details: dict[str, object] | None = None

    if len(bad_channels) == 0:
        quality_mode = "all_channels"
        used_channels = tuple(PREDICTION_QUALITY_CHANNELS)
        threshold_source = "T_all"
        try:
            result = predict_trial(
                trial=trial,
                profile=profile,
            )
            action_score = float(result.action_score)
            action_threshold = float(result.action_threshold)

            channel_powers = {
                str(channel): float(value)
                for channel, value in result.channel_powers.items()
            }

            erd_values = {
                str(channel): float(value)
                for channel, value in result.channel_erd.items()
            }
            roi_details = dict(result.action_score_details)

            channel_baselines = {
                channel: float(profile.channel_baselines[channel])
                for channel in used_channels
            }

            stage2_result = result.stage2_result
            laterality_score = result.laterality_score
            laterality_threshold = result.laterality_threshold
            is_action = result.stage1_result == ACTION_LABEL
        except Exception as exc:
            if profile_source == "fallback":
                return {
                    "success": False,
                    "code": -1,
                    "status": "invalid_fallback_profile",
                    "message": str(exc),
                    "fallback_profile_path": str(profile_path),
                }

            return {
                "success": False,
                "code": -1,
                "status": "invalid_personal_profile",
                "message": str(exc),
            }
        logger.info(
            "prediction window accepted: mode=all_channels, used_channels=6, threshold=T_all"
        )
    else:
        excluded_channel = bad_channels[0]
        action_threshold = get_threshold_without_channel(
            profile,
            excluded_channel,
        )
        if action_threshold is None:
            logger.info(
                "prediction window invalid: code=-1, reason=missing_single_channel_threshold, "
                "excluded_channel=%s, profile_requires_recalibration=true",
                excluded_channel,
            )
            return invalid_quality(
                "missing_single_channel_threshold",
                bad_channels,
                dict(quality_result),
            )
        used_channels = tuple(
            channel for channel in PREDICTION_QUALITY_CHANNELS
            if channel != excluded_channel
        )
        threshold_source = f"T_no_{excluded_channel}"
        quality_mode = "single_channel_excluded"
        try:
            channel_powers = compute_trial_channel_powers(
                trial,
                channels=used_channels,
            )
            channel_baselines = {
                channel: float(profile.channel_baselines[channel])
                for channel in used_channels
            }
            erd_values = compute_channel_erd(
                channel_powers,
                channel_baselines,
            )
            roi_details = dict(
                compute_action_score_details(
                    erd_values,
                    channels=used_channels,
                )
            )
            action_score = float(roi_details["action_score"])
        except Exception as exc:
            if profile_source == "fallback":
                return {
                    "success": False,
                    "code": -1,
                    "status": "invalid_fallback_profile",
                    "message": str(exc),
                    "fallback_profile_path": str(profile_path),
                }

            return {
                "success": False,
                "code": -1,
                "status": "invalid_personal_profile",
                "message": str(exc),
            }
        is_action = action_score >= action_threshold
        stage2_result = None
        laterality_score = None
        laterality_threshold = profile.laterality_threshold
        logger.info(
            "prediction window degraded: excluded_channel=%s, used_channels=5, threshold=%s",
            excluded_channel,
            threshold_source,
        )

    # T3：本次预测计算完成。
    prediction_result_epoch = time.time()
    prediction_result_perf = time.perf_counter()

    prediction_index = int(
        session.prediction_trial_count
    )

    window_start_sample = int(
        (prediction_index - 1) * step_samples
    )

    window_end_sample = int(
        window_start_sample + trial_samples
    )

    window_start_s = float(
        window_start_sample / rate
    )

    window_end_s = float(
        window_end_sample / rate
    )

    prediction_compute_ms = float(
        (
                prediction_result_perf
                - prediction_window_ready_perf
        )
        * 1000.0
    )

    score_margin = float(
        action_score - action_threshold
    )

    label = (
        ACTION_LABEL
        if is_action
        else REST_LABEL
    )

    report_summary = build_prediction_report_summary(
        powers=channel_powers,
        baselines=channel_baselines,
        channel_erd=erd_values,
        roi_details=roi_details or {},
        used_channels=used_channels,
        bad_channels=bad_channels,
        quality_mode=quality_mode,
        threshold_source=threshold_source,
        action_score=action_score,
        action_threshold=action_threshold,
        score_margin=score_margin,
        backend_trial_count=backend_trial_count_for_trace,
        trial_prediction_index=trial_prediction_index,
        prediction_index=prediction_index,
        algorithm_window_result="Action / MI" if is_action else "Rest",
    )
    decision_detail = dict(report_summary.get("decision_detail", {}) or {})

    response: dict[str, object] = {
        "success": True,
        "code": PREDICTION_ACTION if is_action else PREDICTION_REST,
        "status": "predicted",
        "classification_available": True,
        "quality_mode": quality_mode,
        "used_channels": list(used_channels),
        "bad_channels": bad_channels,
        "quality_result": dict(quality_result),
        "threshold_source": threshold_source,
        "case_no": str(case_no),
        "profile_id": profile.profile_id,
        "profile_source": profile_source,
        "calibration_bypassed": (
            calibration_bypassed
        ),
        "label": label,
        "stage1_result": label,
        "stage2_result": None,
        "final_result": label,
        "action_score": float(
            action_score
        ),
        "action_threshold": float(
            action_threshold
        ),
        # =========================
        # 当前预测窗口身份
        # =========================

        "prediction_index": int(
            prediction_index
        ),

        "profile_path": str(
            profile_path
        ),

        # =========================
        # 当前预测窗口位置
        # =========================

        "window_start_sample": int(
            window_start_sample
        ),

        "window_end_sample": int(
            window_end_sample
        ),

        "window_start_s": float(
            window_start_s
        ),

        "window_end_s": float(
            window_end_s
        ),

        # =========================
        # 当前窗口各通道Power
        # =========================

        "channel_powers": {
            str(channel): float(value)
            for channel, value
            in channel_powers.items()
        },

        # =========================
        # Profile里的个人静息Baseline
        # =========================

        "channel_baselines": {
            str(channel): float(value)
            for channel, value
            in channel_baselines.items()
        },

        # =========================
        # 当前窗口各通道ERD
        # =========================

        "channel_erd": {
            str(channel): float(value)
            for channel, value
            in erd_values.items()
        },

        # =========================
        # 当前Score距离实际使用阈值
        # =========================

        "score_margin": float(
            score_margin
        ),
        "report_summary": report_summary,
        "decision_detail": decision_detail,

        # =========================
        # 当前窗口实际预测计算耗时
        # =========================

        "prediction_compute_ms": float(
            prediction_compute_ms
        ),

        "laterality_score": None if laterality_score is None else float(laterality_score),
        "laterality_threshold": float(laterality_threshold),
        "buffer_samples": int(
            session.prediction_buffer.shape[1]
        ),
        "required_samples": trial_samples,
        "prediction_step_samples": (
            step_samples
        ),
        # 当前窗口准备预测的时间。
        "prediction_window_ready_epoch": float(
            prediction_window_ready_epoch
        ),

        # 当前窗口准备预测的高精度计时点。
        "prediction_window_ready_perf": float(
            prediction_window_ready_perf
        ),

        # 当前窗口预测计算完成的时间。
        "prediction_result_epoch": float(
            prediction_result_epoch
        ),

        # 当前窗口预测计算完成的高精度计时点。
        "prediction_result_perf": float(
            prediction_result_perf
        ),
        "message": (
            "action detected"
            if is_action
            else "rest detected"
        ),
    }

    if profile_source == "fallback":
        response[
            "fallback_profile_path"
        ] = str(profile_path)

    eeg_trace.trace_prediction_window_summary(
        backend_trial_count=backend_trial_count_for_trace,
        trial_prediction_index=trial_prediction_index,
        prediction_index=prediction_index,
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        sample_rate=rate,
        channel_count=len(REQUIRED_CHANNELS),
        sample_count=trial_samples,
        quality_mode=quality_mode,
        used_channels=used_channels,
        bad_channels=bad_channels,
        channel_powers=channel_powers,
        channel_baselines=channel_baselines,
        channel_erd=erd_values,
        roi_details=roi_details,
        threshold_source=threshold_source,
        action_score=action_score,
        action_threshold=action_threshold,
        score_margin=score_margin,
        preprocess_summary=getattr(trial, "preprocess_summary", {}),
        code=int(response["code"]),
    )

    return response
