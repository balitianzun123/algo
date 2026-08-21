"""Streaming raw-EEG adapter for configurable personal Rest-N calibration."""

from __future__ import annotations

import importlib
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import time
import numpy as np

from algo.erd.algorithm_core.calibration import calibrate_rest, validate_required_rest_trials
from algo.erd.algorithm_core.constants import DEFAULT_REQUIRED_REST_TRIALS, REQUIRED_CHANNELS, REST_LABEL, ACTION_CHANNELS
from algo.erd.algorithm_core.exceptions import InvalidRequiredRestTrialsError
from algo.erd.algorithm_core.models import PersonalProfile, RawTrialInput, TrialInput
from algo.erd.algorithm_core.preprocessing import preprocess_raw_trial
from algo.erd.algorithm_core.profile_io import save_profile
from algo.erd.algorithm_core.features import compute_trial_channel_powers
from algo.erd.algorithm_core import trace as eeg_trace


DEFAULT_PROFILE_DIR = Path(r"algo/erd/profiles")
TRIAL_SECONDS = 4.0
REST10_REQUIRED_TRIALS = DEFAULT_REQUIRED_REST_TRIALS
REST_BUS_TYPE = 0
CALIBRATION_TYPE = 0
PREDICTION_TYPE = 1

# 后端 ADC 原始码转换：无符号中点移到 0，再换算为浮点信号值。
RAW_ADC_OFFSET = 8388608.0
RAW_ADC_SCALE = 0.0223517
MIN_PROCESSING_SAMPLE_RATE_HZ = 100.0
CALIBRATION_QUALITY_CHANNELS: tuple[str, ...] = ("FC3", "FC4", "C3", "C4", "CP3", "CP4")
CALIBRATION_ACCEPTED_QUALITY_STATUSES = frozenset({"good", "acceptable"})
CALIBRATION_QUALITY_WINDOW_S: tuple[float, float] = (0, 4)

logger = logging.getLogger(__name__)
_impedance_module: Any | None = None
_impedance_import_error: str | None = None


@dataclass
class Rest10StreamSession:
    """Per-case raw stream buffers and the immutable active Rest-N setting."""

    case_no: str
    age: int | None = None
    gender: int | None = None
    required_rest_trials: int = DEFAULT_REQUIRED_REST_TRIALS
    calibration_buffer: np.ndarray | None = None
    prediction_buffer: np.ndarray | None = None
    prediction_window_samples: int | None = None
    prediction_step_samples: int | None = None
    calibration_sample_rate: float | None = None
    prediction_sample_rate: float | None = None
    rest_trials: list[TrialInput] = field(default_factory=list)
    calibration_trial_count: int = 0
    rejected_trial_count: int = 0
    last_trial_quality: dict[str, object] | None = None
    prediction_trial_count: int = 0
    prediction_backend_trial_count: int | None = None
    trial_prediction_index: int = 0
    is_calibrated: bool = False
    profile_id: str | None = None


_SESSIONS: dict[str, Rest10StreamSession] = {}


def reset_session(case_no: str | None = None) -> None:
    """Clear in-memory buffers only; saved caller profiles are never deleted."""
    eeg_trace.reset_session_trace(case_no)
    if case_no is None:
        _SESSIONS.clear()
    else:
        _SESSIONS.pop(str(case_no), None)


def get_session(
    case_no: str, *, age: int | None = None, gender: int | None = None,
    required_rest_trials: int = DEFAULT_REQUIRED_REST_TRIALS,
) -> Rest10StreamSession:
    """Return or create one case session without changing an active setting."""
    key = str(case_no)
    session = _SESSIONS.get(key)
    if session is None:
        session = Rest10StreamSession(case_no=key, age=age, gender=gender, required_rest_trials=required_rest_trials)
        _SESSIONS[key] = session
    else:
        if age is not None:
            session.age = int(age)
        if gender is not None:
            session.gender = int(gender)
    return session


def profile_path_for_case(
    case_no: str, profile_dir: str | Path = DEFAULT_PROFILE_DIR,
    *, required_rest_trials: int = DEFAULT_REQUIRED_REST_TRIALS,
) -> Path:
    """Return the compatibility-safe `<case>_rest<N>_profile.json` path."""
    count = validate_required_rest_trials(required_rest_trials)
    safe_case = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(case_no)).strip("_") or "case"
    return Path(profile_dir) / f"{safe_case}_rest{count}_profile.json"


def validate_stream_sample_rate(sample_rate: object) -> float:
    """Validate a rate that can represent the fixed 50 Hz preprocessing."""
    if isinstance(sample_rate, (bool, np.bool_)):
        raise ValueError("sample_rate must be a positive numeric value, not bool.")
    try:
        rate = float(sample_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("sample_rate must be a positive numeric value.") from exc
    if not np.isfinite(rate) or rate <= MIN_PROCESSING_SAMPLE_RATE_HZ:
        raise ValueError(
            f"sample_rate must be greater than {MIN_PROCESSING_SAMPLE_RATE_HZ:g} Hz for 50 Hz preprocessing, got {sample_rate!r}."
        )
    return rate


def convert_backend_raw_samples(filtered_samples: np.ndarray) -> np.ndarray:
    """Convert one backend raw chunk before buffering or signal preprocessing.

    The adapter expects unsigned 24-bit ADC codes. Conversion is performed
    once per incoming chunk and on a copy, so caller-owned arrays are intact.
    """
    samples = np.asarray(filtered_samples, dtype=np.float64).copy()
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError(f"filtered_samples must be nonempty [time_steps, channels], got {samples.shape}.")
    if not np.isfinite(samples).all():
        raise ValueError("filtered_samples contains NaN or Inf.")
    samples -= RAW_ADC_OFFSET
    samples *= RAW_ADC_SCALE
    if not np.isfinite(samples).all():
        raise ValueError("converted backend samples contain NaN or Inf.")
    return samples


def backend_samples_to_required_channels(filtered_samples: np.ndarray, standard_mapping: Mapping[str, int]) -> np.ndarray:
    """Map backend raw `[time, channels]` samples into fixed `[8, time]` order."""
    # 这里只做通道选择和重排，原始数据先进入流式缓存，不在小块上滤波。
    # 转换先于通道映射和流式缓存，校准、预测共用同一数据单位。
    samples = convert_backend_raw_samples(filtered_samples)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError(f"filtered_samples must be nonempty [time_steps, channels], got {samples.shape}.")
    if not np.isfinite(samples).all():
        raise ValueError("filtered_samples contains NaN or Inf.")
    mapping = {str(name): int(index) for name, index in dict(standard_mapping).items()}
    missing = [channel for channel in REQUIRED_CHANNELS if channel not in mapping]
    if missing:
        eeg_trace.trace_block(
            "SESSION",
            [
                "Status      : CHANNEL_MAPPING_ERROR",
                f"Input shape : {tuple(samples.shape)}",
                f"Missing     : {' '.join(missing)}",
            ],
        )
        eeg_trace.trace_channel_map(
            input_shape=samples.shape,
            output_shape=(0, 0),
            selected_channels=REQUIRED_CHANNELS,
            missing_channels=missing,
        )
        raise ValueError(f"STANDARD_MAPPING is missing required channel(s): {missing}.")
    indices = [mapping[channel] for channel in REQUIRED_CHANNELS]
    if min(indices) >= 0 and max(indices) < samples.shape[1]:
        mapped = samples[:, indices].T.copy()
        eeg_trace.trace_channel_map(
            input_shape=samples.shape,
            output_shape=mapped.shape,
            selected_channels=REQUIRED_CHANNELS,
            missing_channels=(),
        )
        return mapped
    ordered_names = [str(name) for name in standard_mapping]
    if samples.shape[1] >= len(ordered_names) and all(channel in ordered_names for channel in REQUIRED_CHANNELS):
        by_position = {name: index for index, name in enumerate(ordered_names)}
        mapped = samples[:, [by_position[channel] for channel in REQUIRED_CHANNELS]].T.copy()
        eeg_trace.trace_channel_map(
            input_shape=samples.shape,
            output_shape=mapped.shape,
            selected_channels=REQUIRED_CHANNELS,
            missing_channels=(),
        )
        return mapped
    eeg_trace.trace_block(
        "SESSION",
        [
            "Status      : CHANNEL_MAPPING_ERROR",
            f"Input shape : {tuple(samples.shape)}",
            f"Mapping     : incompatible with STANDARD_MAPPING",
        ],
    )
    raise ValueError(f"filtered_samples is incompatible with STANDARD_MAPPING: shape={samples.shape}.")


def append_stream_buffer(buffer: np.ndarray | None, new_samples_8xt: np.ndarray) -> np.ndarray:
    """Append mapped raw samples without filtering individual backend blocks."""
    samples = np.asarray(new_samples_8xt, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] != len(REQUIRED_CHANNELS):
        raise ValueError(f"stream samples must have shape [8, time_steps], got {samples.shape}.")
    return samples.copy() if buffer is None else np.concatenate([np.asarray(buffer, dtype=np.float64), samples], axis=1)


def pop_trial_from_buffer(buffer: np.ndarray, trial_samples: int) -> tuple[np.ndarray | None, np.ndarray]:
    """Remove one complete raw trial from a stream buffer, if available."""
    data = np.asarray(buffer, dtype=np.float64)
    if data.shape[1] < int(trial_samples):
        return None, data
    return data[:, :trial_samples].copy(), data[:, trial_samples:].copy()


def make_preprocessed_trial_input(
    *, case_no: str, role: str, index: int, raw_data_8x_samples: np.ndarray,
    sample_rate: int | float, label: str | None,
) -> TrialInput:
    """Preprocess one complete raw four-second trial exactly once.

    The stream adapter guarantees that sample zero is the task event start;
    therefore `event_sample=0` is the public streaming-window contract.
    """
    # 完整 4 秒窗口的第一个采样点就是事件起点，保持 event_sample=0 约定。
    raw_trial = RawTrialInput(
        trial_id=f"{case_no}_{role}_{index:03d}", data=np.asarray(raw_data_8x_samples, dtype=np.float64),
        channel_names=tuple(REQUIRED_CHANNELS), sampling_rate=float(sample_rate),
        data_unit="unknown_raw_npy_units_preserved", label=label, event_sample=0, subject_id=str(case_no),
    )
    return preprocess_raw_trial(raw_trial)


def _load_impedance_module() -> Any | None:
    global _impedance_module, _impedance_import_error
    if _impedance_module is not None:
        return _impedance_module
    try:
        _impedance_module = importlib.import_module("impedance")
        _impedance_import_error = None
    except Exception as exc:
        _impedance_import_error = str(exc)
        logger.exception("failed to load impedance module")
        return None
    logger.info(
        "loaded impedance module from %s",
        getattr(_impedance_module, "__file__", "<built-in>"),
    )
    return _impedance_module


def _rest_trial_quality_failure(error: str) -> dict[str, object]:
    return {
        "passed": False,
        "channels": [],
        "failed_channels": list(CALIBRATION_QUALITY_CHANNELS),
        "error": str(error),
    }


def evaluate_rest_trial_quality(trial: TrialInput) -> dict[str, object]:
    """Evaluate the current preprocessed rest trial without touching live quality state."""
    try:
        data = np.asarray(trial.data, dtype=np.float64)
        channel_names = tuple(str(name) for name in trial.channel_names)
        if data.ndim != 2:
            raise ValueError(f"trial data must have shape [channels, samples], got {data.shape}.")
        if data.shape[0] != len(channel_names):
            raise ValueError("trial channel dimension must match channel_names length.")
        missing = [channel for channel in CALIBRATION_QUALITY_CHANNELS if channel not in channel_names]
        if missing:
            return {
                "passed": False,
                "channels": [],
                "failed_channels": missing,
                "error": f"missing calibration quality channel(s): {missing}",
            }
        rate = float(trial.sampling_rate)
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError(f"sampling_rate must be positive and finite, got {trial.sampling_rate}.")
        start_sample = int(round(CALIBRATION_QUALITY_WINDOW_S[0] * rate))
        end_sample = int(round(CALIBRATION_QUALITY_WINDOW_S[1] * rate))
        if start_sample < 0 or end_sample <= start_sample or end_sample > data.shape[1]:
            raise ValueError(
                f"quality window samples [{start_sample}, {end_sample}) exceed trial shape {data.shape}."
            )
        if not np.isfinite(data).all():
            raise ValueError("trial data contains NaN or Inf.")

        name_to_index = {name: index for index, name in enumerate(channel_names)}
        indices = [name_to_index[channel] for channel in CALIBRATION_QUALITY_CHANNELS]
        quality_data = data[indices, start_sample:end_sample]
        impedance_module = _load_impedance_module()
        if impedance_module is None:
            raise RuntimeError(_impedance_import_error or "impedance module is not available")
        quality = impedance_module.eeg_signal_impedance(
            quality_data,
            rate,
            list(CALIBRATION_QUALITY_CHANNELS),
        )
        if not isinstance(quality, Mapping):
            raise ValueError("quality result must be a dict-like object.")
        channel_eval = quality.get("channel_eval")
        if not isinstance(channel_eval, (list, tuple)):
            raise ValueError("quality result is missing channel_eval.")

        status_by_name = {name: "unknown" for name in CALIBRATION_QUALITY_CHANNELS}
        for index, name in enumerate(CALIBRATION_QUALITY_CHANNELS):
            item = channel_eval[index] if index < len(channel_eval) and isinstance(channel_eval[index], Mapping) else {}
            item_name = str(item.get("name", name))
            target_name = item_name if item_name in status_by_name else name
            status_by_name[target_name] = str(item.get("status", "unknown"))

        channels = [
            {"name": name, "status": status_by_name[name]}
            for name in CALIBRATION_QUALITY_CHANNELS
        ]
        failed_channels = [
            item["name"]
            for item in channels
            if item["status"] not in CALIBRATION_ACCEPTED_QUALITY_STATUSES
        ]
        return {
            "passed": not failed_channels,
            "channels": channels,
            "failed_channels": failed_channels,
        }
    except Exception as exc:
        return _rest_trial_quality_failure(str(exc))


def _invalid_count_result(exc: Exception) -> dict[str, object]:
    return {"success": False, "code": -1, "status": "invalid_required_rest_trials", "message": str(exc)}


def _calibration_started(session: Rest10StreamSession) -> bool:
    return bool(session.rest_trials) or session.calibration_buffer is not None or session.is_calibrated

def train_rest10_stream(
    *, case_no: str, age: int, gender: int, sample_rate: int, type: int, bus_type: int,
    filtered_samples: np.ndarray, standard_mapping: Mapping[str, int], profile_dir: str | Path = DEFAULT_PROFILE_DIR,
    required_rest_trials: int = DEFAULT_REQUIRED_REST_TRIALS,
) -> dict[str, object]:
    """Buffer raw rest data and create a validated Rest-N profile when complete."""
    """ Bridge以1秒 block 向 train/predict 适配层送数据；真正算法 trial/window 是4秒 """
    try:
        count = validate_required_rest_trials(required_rest_trials)
    except InvalidRequiredRestTrialsError as exc:
        return _invalid_count_result(exc)
    try:
        rate = validate_stream_sample_rate(sample_rate)
    except ValueError as exc:
        return {"success": False, "code": -1, "status": "invalid_sample_rate", "message": str(exc)}
    if int(type) != CALIBRATION_TYPE:
        return {"success": False, "code": -1, "status": "invalid_type", "message": "train_rest10_stream expects type=0 calibration."}
    if int(bus_type) != REST_BUS_TYPE:
        return {"success": False, "code": -1, "status": "invalid_bus_type", "message": "Rest calibration accepts only bus_type=0 rest data."}
    # session 保存本次校准的 trial 配置，已开始后不允许静默切换数量。
    session = get_session(case_no, age=age, gender=gender, required_rest_trials=count)
    if _calibration_started(session) and session.calibration_sample_rate is not None and not np.isclose(session.calibration_sample_rate, rate):
        return {
            "success": False, "code": -1, "status": "session_configuration_mismatch",
            "message": "sample_rate changed during an active calibration session; reset the session before starting a new calibration.",
        }
    if _calibration_started(session) and session.required_rest_trials != count:
        return {
            "success": False, "code": -1, "status": "session_configuration_mismatch",
            "message": "required_rest_trials changed during an active calibration session; reset the session before starting a new calibration.",
        }
    if not _calibration_started(session):
        session.required_rest_trials = count
        session.calibration_sample_rate = rate
    trial_samples = int(round(rate * TRIAL_SECONDS))
    eeg_trace.trace_session(
        case_no=str(case_no),
        mode="calibration",
        sample_rate=rate,
        input_channels=int(np.asarray(filtered_samples).shape[1]) if np.asarray(filtered_samples).ndim == 2 else 0,
        required_channels=REQUIRED_CHANNELS,
        action_channels=ACTION_CHANNELS,
        window_seconds=TRIAL_SECONDS,
        step_seconds=None,
        mapping=standard_mapping,
    )
    session.calibration_buffer = append_stream_buffer(
        session.calibration_buffer, backend_samples_to_required_channels(filtered_samples, standard_mapping)
    )
    """重点看这里"""
    # 只有累计满一个 trial 才预处理，避免每个后端小块产生滤波边界效应。
    last_trial_response: dict[str, object] | None = None
    # 累计10个有效trial才行
    while len(session.rest_trials) < session.required_rest_trials:
        raw_trial, session.calibration_buffer = pop_trial_from_buffer(session.calibration_buffer, trial_samples)
        if raw_trial is None:
            break
        session.calibration_trial_count += 1
        trial_ready_epoch = time.time()
        trial_ready_perf = time.perf_counter()

        trial_channel_powers: dict[str, float] = {}
        trial_power_error: str | None = None
        try:
            trial_input = make_preprocessed_trial_input(
                case_no=session.case_no, role="rest_cal", index=session.calibration_trial_count,
                raw_data_8x_samples=raw_trial, sample_rate=rate, label=REST_LABEL,
            )
            quality_result = evaluate_rest_trial_quality(trial_input)
            try:
                trial_channel_powers = (compute_trial_channel_powers(trial_input,channels=ACTION_CHANNELS,)
                )
            except Exception as exc:
                trial_power_error = str(exc)
        except Exception as exc:
            trial_input = None
            quality_result = _rest_trial_quality_failure(str(exc))

        if not isinstance(quality_result, Mapping):
            quality_result = _rest_trial_quality_failure("quality result must be a dict-like object.")
        failed_channels = [str(channel) for channel in quality_result.get("failed_channels", [])]
        quality_passed = bool(quality_result.get("passed")) and not failed_channels
        session.last_trial_quality = dict(quality_result)
        attempt = int(session.calibration_trial_count)

        # ==========================================
        # 当前4秒Calibration Trial诊断信息
        # ==========================================

        trial_id = (
            None
            if trial_input is None
            else str(trial_input.trial_id)
        )

        # Calibration trial是非重叠4秒trial。
        # attempt=1 → 0~4s
        # attempt=2 → 4~8s
        # attempt=3 → 8~12s
        trial_start_sample = int((attempt - 1) * trial_samples)

        trial_end_sample = int(trial_start_sample + trial_samples)

        trial_start_s = float(trial_start_sample / rate)

        trial_end_s = float(trial_end_sample / rate)

        # 当前完整trial的算法处理完成时间。
        trial_result_epoch = time.time()
        trial_result_perf = time.perf_counter()

        trial_compute_ms = float((trial_result_perf - trial_ready_perf)* 1000.0)

        if quality_passed and trial_input is not None:
            session.rest_trials.append(trial_input)
            accepted = len(session.rest_trials)
            last_trial_response = {
                "success": True, "code": 0, "status": "rest_trial_accepted", "case_no": session.case_no,
                "collected_trials": accepted, "accepted_trials": accepted,
                "rejected_trials": session.rejected_trial_count, "attempts": attempt,
                "required_trials": session.required_rest_trials,
                "buffer_samples": int(session.calibration_buffer.shape[1]),
                "trial_quality": session.last_trial_quality,
                # 当前是本轮第几个实际尝试的4秒trial。
                "trial_index": int(attempt),
                # TrialInput自己的trial_id。
                "trial_id": trial_id,
                # 当前trial对应Calibration数据流的位置。
                "trial_start_sample": int(trial_start_sample),
                "trial_end_sample": int(trial_end_sample),
                "trial_start_s": float(trial_start_s),
                "trial_end_s": float(trial_end_s),
                # 当前质量算法实际使用的trial内部时间窗。
                "quality_window_s": [
                    float(
                        CALIBRATION_QUALITY_WINDOW_S[0]
                    ),
                    float(
                        CALIBRATION_QUALITY_WINDOW_S[1]
                    ),
                ],

                # 当前trial长度。
                "trial_samples": int(trial_samples),
                "trial_seconds": float(TRIAL_SECONDS),
                # 当前trial六个动作/质量通道的8-30Hz功率。
                "trial_channel_powers": dict(trial_channel_powers),
                # Power只是诊断，失败不影响trial accepted/rejected。
                "trial_power_error": (trial_power_error),
                # 当前trial算法处理时间。
                "trial_ready_epoch": float(trial_ready_epoch),
                "trial_result_epoch": float(trial_result_epoch),
                "trial_compute_ms": float(trial_compute_ms),

                "message": "Rest calibration trial accepted.",
            }
            logger.info(
                "rest trial accepted: attempt=%d, accepted=%d/%d, rejected=%d",
                attempt,
                accepted,
                session.required_rest_trials,
                session.rejected_trial_count,
            )
            eeg_trace.trace_calibration_trial(
                attempt=attempt,
                accepted=True,
                quality_result=quality_result,
                action_channels=ACTION_CHANNELS,
                bad_channels=failed_channels,
                accepted_count=accepted,
                required_count=session.required_rest_trials,
                trial_start_s=trial_start_s,
                trial_end_s=trial_end_s,
                trial_powers=trial_channel_powers,
                trial_power_error=trial_power_error,
            )
        else:
            session.rejected_trial_count += 1
            if not failed_channels:
                failed_channels = list(CALIBRATION_QUALITY_CHANNELS)
                session.last_trial_quality["failed_channels"] = failed_channels
            accepted = len(session.rest_trials)
            last_trial_response = {
                "success": True, "code": 0, "status": "rest_trial_rejected", "case_no": session.case_no,
                "collected_trials": accepted, "accepted_trials": accepted,
                "rejected_trials": session.rejected_trial_count, "attempts": attempt,
                "required_trials": session.required_rest_trials,
                "buffer_samples": int(session.calibration_buffer.shape[1]),
                "failed_channels": failed_channels,
                "trial_quality": session.last_trial_quality,

                "trial_index": int(attempt),

                "trial_id": trial_id,

                "trial_start_sample": int(trial_start_sample),

                "trial_end_sample": int(trial_end_sample),

                "trial_start_s": float(trial_start_s),

                "trial_end_s": float(trial_end_s),

                "quality_window_s": [
                    float(
                        CALIBRATION_QUALITY_WINDOW_S[0]
                    ),
                    float(
                        CALIBRATION_QUALITY_WINDOW_S[1]
                    ),
                ],

                "trial_samples": int(trial_samples),

                "trial_seconds": float(TRIAL_SECONDS),

                "trial_channel_powers": dict(trial_channel_powers),

                "trial_power_error": (trial_power_error),

                "trial_ready_epoch": float(trial_ready_epoch),

                "trial_result_epoch": float(trial_result_epoch),

                "trial_compute_ms": float(trial_compute_ms),

                "message": "Rest calibration trial rejected; collecting replacement trial.",
            }
            logger.info(
                "rest trial rejected: attempt=%d, accepted=%d/%d, rejected=%d, "
                "failed_channels=%s, quality_error=%s, quality_channels=%s",
                attempt,
                accepted,
                session.required_rest_trials,
                session.rejected_trial_count,
                failed_channels,
                session.last_trial_quality.get("error"),
                session.last_trial_quality.get("channels"),
            )
            eeg_trace.trace_calibration_trial(
                attempt=attempt,
                accepted=False,
                quality_result=quality_result,
                action_channels=ACTION_CHANNELS,
                bad_channels=failed_channels,
                accepted_count=accepted,
                required_count=session.required_rest_trials,
                trial_start_s=trial_start_s,
                trial_end_s=trial_end_s,
                trial_powers=trial_channel_powers,
                trial_power_error=trial_power_error,
            )
    if len(session.rest_trials) < session.required_rest_trials:
        if last_trial_response is not None:
            return last_trial_response
        return {
            "success": True, "code": 0, "status": "calibrating", "case_no": session.case_no,
            "collected_trials": len(session.rest_trials), "required_trials": session.required_rest_trials,
            "accepted_trials": len(session.rest_trials), "rejected_trials": session.rejected_trial_count,
            "attempts": session.calibration_trial_count, "last_trial_quality": session.last_trial_quality,
            "buffer_samples": int(session.calibration_buffer.shape[1]), "message": "Rest calibration in progress.",
        }

    # 10个有效Rest Trial收齐后，
    # 开始建立个人Baseline和Threshold Profile
    profile_build_started_epoch = time.time()
    profile_build_started_perf = (time.perf_counter())


    profile: PersonalProfile = calibrate_rest(
        # 校准核心只接收已经完成统一预处理的 TrialInput。
        subject_id=session.case_no, rest_trials=tuple(session.rest_trials[:session.required_rest_trials]),
        channel_names=tuple(REQUIRED_CHANNELS), sampling_rate=rate,
        required_rest_trials=session.required_rest_trials,
    )
    path = profile_path_for_case(session.case_no, profile_dir, required_rest_trials=session.required_rest_trials)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_profile(profile, path, overwrite=True)

    profile_build_finished_epoch = time.time()
    profile_build_finished_perf = (time.perf_counter())

    profile_build_ms = float((profile_build_finished_perf- profile_build_started_perf) * 1000.0)
    session.is_calibrated, session.profile_id = True, profile.profile_id
    logger.info(
        "rest calibration completed: accepted=%d, rejected=%d, attempts=%d",
        session.required_rest_trials,
        session.rejected_trial_count,
        session.calibration_trial_count,
    )
    eeg_trace.trace_calibration_summary(
        accepted_trials=session.required_rest_trials,
        rejected_trials=session.rejected_trial_count,
        attempts=session.calibration_trial_count,
        action_threshold=float(profile.action_threshold),
        thresholds_without_channel=dict(profile.action_thresholds_without_channel),
        baselines=dict(profile.channel_baselines),
        threshold_source=str(getattr(profile, "threshold_source_action", "")),
    )
    return {
        "success": True, "code": 0, "status": "profile_created", "case_no": session.case_no,
        "profile_id": profile.profile_id, "profile_path": str(path),
        "collected_trials": session.required_rest_trials, "required_trials": session.required_rest_trials,
        "accepted_trials": session.required_rest_trials, "rejected_trials": session.rejected_trial_count,
        "attempts": session.calibration_trial_count, "last_trial_quality": session.last_trial_quality,
        "final_trial": (last_trial_response),"profile_build_started_epoch": float(profile_build_started_epoch),
        "profile_build_finished_epoch": float(profile_build_finished_epoch),"profile_build_ms": float(profile_build_ms),
        "profile_summary": {"channel_baselines": {
                str(channel): float(value)
                for channel, value
                in profile.channel_baselines.items()
            },

            "action_threshold": float(
                profile.action_threshold
            ),

            "action_thresholds_without_channel": {
                str(channel): float(value)
                for channel, value in dict(getattr(profile,"action_thresholds_without_channel", {},)or {}).items()
            },

            "threshold_source_action": str(getattr(profile,"threshold_source_action","",)),
            "calibration_power_details": dict(
                profile.extra.get("calibration_power_details", {})
            ),
            "criterion_details": dict(
                profile.extra.get("criterion_details", {})
            ),
        },
        "message": "Rest calibration completed.",
    }
