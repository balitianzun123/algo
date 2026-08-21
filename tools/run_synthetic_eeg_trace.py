"""Run one synthetic EEG calibration + prediction trace session.

The generated data is virtual/synthetic only. It is shaped like the backend
protocol: unsigned 24-bit ADC counts with shape [time, 16 channels].
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from algo.erd.algorithm_bridge import EEGAlgorithmBridge
from algo.erd.algorithm_core import trace as eeg_trace
from algo.erd.algorithm_core.reporting import (
    format_calibration_final_summary_report,
    format_calibration_loo_report,
    format_prediction_success_window_summary,
    format_window_identity_for_report,
)
from algo.erd.enter import ALGORITHM_RUNTIME_CONFIG, DEVICE_STANDARD_MAPPING
from algo.erd.train import RAW_ADC_OFFSET, RAW_ADC_SCALE, profile_path_for_case, reset_session


SYNTHETIC_SEED = 20260817
CASE_NO = "SYNTH_TRACE_001"
AGE = 60
GENDER = 1
SAMPLE_RATE = 2000
INPUT_CHANNELS = 16
REQUIRED_REST_TRIALS = int(ALGORITHM_RUNTIME_CONFIG.get("required_rest_trials", 10))
TRIAL_SECONDS = 4.0
PREDICTION_STEP_SECONDS = float(ALGORITHM_RUNTIME_CONFIG.get("prediction_step_seconds", 1.0))
PREDICTION_SECONDS = 12.0

ARTIFACT_DIR = REPO_ROOT / "artifacts" / "synthetic_trace"
CALIBRATION_TRACE_PATH = ARTIFACT_DIR / "synthetic_calibration_trace.log"
PREDICTION_TRACE_PATH = ARTIFACT_DIR / "synthetic_prediction_trace.log"
BUSINESS_REPORT_PATH = ARTIFACT_DIR / "synthetic_prediction_business_report.txt"
METADATA_PATH = ARTIFACT_DIR / "synthetic_run_metadata.json"
README_PATH = ARTIFACT_DIR / "README.md"

ACTION_CHANNELS = ("FC3", "FC4", "C3", "C4", "CP3", "CP4")
BASE_AMPLITUDE_UV = {
    "FC3": 8.5,
    "FC4": 9.0,
    "C3": 10.0,
    "C4": 10.5,
    "CP3": 9.4,
    "CP4": 9.8,
    "CPz": 8.2,
    "Cz": 8.0,
}
CHANNEL_PHASE = {
    "FC3": 0.00,
    "FC4": 0.31,
    "C3": 0.62,
    "C4": 0.93,
    "CP3": 1.24,
    "CPz": 1.55,
    "CP4": 1.86,
    "Cz": 2.17,
}
PREDICTION_SEGMENTS = [
    {"name": "REST_LIKE", "start_s": 0.0, "end_s": 4.0, "left_scale": 1.00},
    {"name": "MODERATE_MI_LIKE", "start_s": 4.0, "end_s": 8.0, "left_scale": 0.62},
    {"name": "STRONGER_MI_LIKE", "start_s": 8.0, "end_s": 12.0, "left_scale": 0.35},
]


def _close_trace_file_handlers() -> None:
    for handler in list(eeg_trace.logger.handlers):
        if isinstance(handler, logging.FileHandler):
            eeg_trace.logger.removeHandler(handler)
            handler.close()
    eeg_trace._FILE_HANDLER_READY = False  # type: ignore[attr-defined]


def _begin_clean_trace_file(path: Path, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    path.write_text(header, encoding="utf-8")
    _close_trace_file_handlers()
    eeg_trace.TRACE_LOG_PATH = path  # type: ignore[attr-defined]
    eeg_trace.reset_session_trace()


def _trace_header(*, profile_source: str | None = None) -> str:
    lines = [
        "=" * 60,
        "SYNTHETIC EEG TRACE",
        "THIS IS NOT REAL EEG DATA.",
        "PURPOSE: ALGORITHM OBSERVABILITY / TRACE REVIEW ONLY.",
    ]
    if profile_source is not None:
        lines.append(f"PROFILE SOURCE: {profile_source}")
    lines.extend(["=" * 60, ""])
    return "\n".join(lines)


def _segment_at(time_s: float) -> dict[str, float | str]:
    for segment in PREDICTION_SEGMENTS:
        if float(segment["start_s"]) <= time_s < float(segment["end_s"]):
            return segment
    return PREDICTION_SEGMENTS[-1]


def _amplitude_for_channel(channel: str, *, scenario: str, time_s: float, trial_index: int) -> float:
    base = float(BASE_AMPLITUDE_UV[channel])
    slow_jitter = 1.0 + 0.025 * math.sin(0.47 * float(trial_index) + CHANNEL_PHASE[channel])
    if scenario == "prediction":
        segment = _segment_at(time_s)
        if channel in {"FC3", "C3", "CP3"}:
            return base * float(segment["left_scale"]) * slow_jitter
    return base * slow_jitter


def _synthetic_adc_batch(
    *,
    rng: np.random.Generator,
    start_s: float,
    duration_s: float,
    scenario: str,
    trial_index: int,
) -> np.ndarray:
    samples = int(round(duration_s * float(SAMPLE_RATE)))
    time_s = start_s + np.arange(samples, dtype=np.float64) / float(SAMPLE_RATE)
    uv = rng.normal(0.0, 0.22, size=(samples, INPUT_CHANNELS))
    for channel, protocol_index in DEVICE_STANDARD_MAPPING.items():
        amplitude = np.asarray(
            [
                _amplitude_for_channel(
                    channel,
                    scenario=scenario,
                    time_s=float(t),
                    trial_index=trial_index,
                )
                for t in time_s
            ],
            dtype=np.float64,
        )
        phase = float(CHANNEL_PHASE[channel])
        alpha = amplitude * np.sin(2.0 * np.pi * 10.0 * time_s + phase)
        beta = 0.18 * amplitude * np.sin(2.0 * np.pi * 18.0 * time_s + phase / 2.0)
        drift = 0.08 * np.sin(2.0 * np.pi * 1.0 * time_s + phase)
        uv[:, int(protocol_index)] += alpha + beta + drift
    adc = RAW_ADC_OFFSET + uv / RAW_ADC_SCALE
    adc = np.clip(adc, 1.0, 16777214.0)
    if not np.isfinite(adc).all():
        raise RuntimeError("synthetic ADC contains NaN or Inf")
    return adc.astype(np.float64)


def _feed_batch(bridge: EEGAlgorithmBridge, batch: np.ndarray) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    for row in np.asarray(batch, dtype=np.float64):
        result = bridge.feed(row)
        if isinstance(result, dict):
            responses.append(dict(result))
    return responses


def _run_calibration(bridge: EEGAlgorithmBridge, rng: np.random.Generator) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    responses: list[dict[str, Any]] = []
    bridge.start_calibration()
    rest_start = 0.0
    for trial_index in range(1, REQUIRED_REST_TRIALS + 6):
        batch = _synthetic_adc_batch(
            rng=rng,
            start_s=rest_start + (trial_index - 1) * TRIAL_SECONDS,
            duration_s=TRIAL_SECONDS,
            scenario="rest",
            trial_index=trial_index,
        )
        responses.extend(_feed_batch(bridge, batch))
        profile_created = [item for item in responses if item.get("status") == "profile_created"]
        if profile_created:
            return profile_created[-1], responses
    raise RuntimeError("synthetic calibration did not create a profile")


def _run_prediction(bridge: EEGAlgorithmBridge, rng: np.random.Generator) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    start_result = bridge.start_prediction()
    if not start_result.get("success"):
        raise RuntimeError(f"start_prediction failed: {start_result}")
    use_result = bridge.start_using_prediction_result()
    if not use_result.get("success"):
        raise RuntimeError(f"start_using_prediction_result failed: {use_result}")
    eeg_trace.trace_block(
        "SESSION",
        [
            "Synthetic prediction segments",
            "REST_LIKE          : 0.0s-4.0s",
            "MODERATE_MI_LIKE   : 4.0s-8.0s",
            "STRONGER_MI_LIKE   : 8.0s-12.0s",
            "Note               : MI-like is a synthetic signal scenario, not real physiology.",
        ],
    )
    responses: list[dict[str, Any]] = []
    step_seconds = PREDICTION_STEP_SECONDS
    chunks = int(round(PREDICTION_SECONDS / step_seconds))
    for chunk_index in range(chunks):
        chunk_start = float(chunk_index) * step_seconds
        batch = _synthetic_adc_batch(
            rng=rng,
            start_s=chunk_start,
            duration_s=step_seconds,
            scenario="prediction",
            trial_index=100 + chunk_index,
        )
        responses.extend(_feed_batch(bridge, batch))
    predicted = [item for item in responses if item.get("status") == "predicted"]
    if not predicted:
        raise RuntimeError("synthetic prediction did not produce any predicted window")
    return predicted, responses


def _load_profile_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_metadata(
    *,
    profile_path: Path,
    calibration_result: dict[str, Any],
    prediction_windows: list[dict[str, Any]],
) -> None:
    payload = {
        "synthetic": True,
        "real_eeg": False,
        "seed": SYNTHETIC_SEED,
        "case_no": CASE_NO,
        "sample_rate": SAMPLE_RATE,
        "input_channels": INPUT_CHANNELS,
        "entry_layer": "EEGAlgorithmBridge.feed -> sim_demo -> train_rest10_stream / predict_rest10_stream",
        "calibration_trial_seconds": TRIAL_SECONDS,
        "required_rest_trials": REQUIRED_REST_TRIALS,
        "prediction_window_seconds": TRIAL_SECONDS,
        "prediction_step_seconds": PREDICTION_STEP_SECONDS,
        "prediction_total_seconds": PREDICTION_SECONDS,
        "prediction_window_count": len(prediction_windows),
        "prediction_segments": PREDICTION_SEGMENTS,
        "profile_path": str(profile_path),
        "calibration_trace_path": str(CALIBRATION_TRACE_PATH),
        "prediction_trace_path": str(PREDICTION_TRACE_PATH),
        "business_report_path": str(BUSINESS_REPORT_PATH),
        "accepted_trials": int(calibration_result.get("accepted_trials", 0)),
        "rejected_trials": int(calibration_result.get("rejected_trials", 0)),
        "attempts": int(calibration_result.get("attempts", 0)),
    }
    METADATA_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_business_report(
    *,
    profile_payload: dict[str, Any],
    prediction_windows: list[dict[str, Any]],
) -> None:
    successful = [
        item
        for item in prediction_windows
        if int(item.get("display_code") or 0) == 1
    ]
    if not successful:
        successful = [
            item
            for item in prediction_windows
            if int(item.get("code") or 0) == 1
        ]
    first_success = successful[0] if successful else {}
    identity = format_window_identity_for_report(first_success)
    text = (
        "THIS FILE IS A SYNTHETIC / VIRTUAL EEG BUSINESS REPORT PREVIEW.\n"
        "IT IS NOT REAL EEG DATA.\n"
        "\n"
        "脑电运动想象预测报告\n"
        "====================\n"
        "\n"
        "校准静息功率基线与MI阈值配置\n"
        "------------------------------\n"
        f"{format_calibration_loo_report(profile_payload)}"
        "\n"
        f"{format_calibration_final_summary_report(profile_payload)}"
        "\n"
    )
    if first_success:
        text += (
            "========================================\n"
            "Prediction Trial #SYNTHETIC\n"
            "========================================\n"
            f"后端 Trial       : {identity.get('backend_trial_count', 'N/A')}\n"
            f"Trial Window     : {identity.get('trial_prediction_index', 'N/A')}\n"
            f"Global Window    : {identity.get('prediction_index', 'N/A')}\n"
            "\n"
            "结果             : 成功\n"
            f"最终 display_code: {first_success.get('display_code')}\n"
            "\n"
            f"{format_prediction_success_window_summary(first_success)}"
            "\n"
            "========================================\n"
        )
    else:
        text += "未产生 synthetic success window。\n"
    BUSINESS_REPORT_PATH.write_text(text, encoding="utf-8")


def _write_readme(*, profile_path: Path, calibration_result: dict[str, Any], prediction_windows: list[dict[str, Any]]) -> None:
    text = f"""# Synthetic EEG Trace Run

This run uses synthetic / virtual EEG-like data only. It is not real EEG data.

- Entry layer: `EEGAlgorithmBridge.feed()` using 16-channel unsigned ADC frames.
- Dispatch path: `feed -> sim_demo -> train_rest10_stream / predict_rest10_stream`.
- Sample rate: {SAMPLE_RATE} Hz.
- Calibration trial length: {TRIAL_SECONDS:.1f} s.
- Accepted Rest trials: {int(calibration_result.get("accepted_trials", 0))}.
- Rejected Rest trials: {int(calibration_result.get("rejected_trials", 0))}.
- Calibration attempts: {int(calibration_result.get("attempts", 0))}.
- Prediction window: {TRIAL_SECONDS:.1f} s.
- Prediction step: {PREDICTION_STEP_SECONDS:.1f} s.
- Prediction windows generated: {len(prediction_windows)}.
- Random seed: {SYNTHETIC_SEED}.
- Profile path: `{profile_path}`.
- Calibration trace path: `{CALIBRATION_TRACE_PATH}`.
- Prediction trace path: `{PREDICTION_TRACE_PATH}`.
- Prediction business report preview: `{BUSINESS_REPORT_PATH}`.
- Metadata path: `{METADATA_PATH}`.

## Synthetic Rest Signal

The calibration input is generated as 16-channel unsigned ADC counts around
`RAW_ADC_OFFSET`. The mapped EEG channels contain deterministic 10 Hz activity,
small 18 Hz content, low-amplitude drift, low random noise, and slight
channel-specific amplitude/phase differences.

## Synthetic Prediction Signal

Prediction uses three virtual signal scenarios:

- `REST_LIKE`: 0.0s-4.0s, close to calibration Rest input.
- `MODERATE_MI_LIKE`: 4.0s-8.0s, lower 10 Hz amplitude on the left ROI action channels.
- `STRONGER_MI_LIKE`: 8.0s-12.0s, stronger reduction on the same ROI.

`MI-like` is only a synthetic signal scenario name. It is not real physiology
and is not an accuracy validation.
"""
    README_PATH.write_text(text, encoding="utf-8")


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = profile_path_for_case(
        CASE_NO,
        ARTIFACT_DIR,
        required_rest_trials=REQUIRED_REST_TRIALS,
    )
    for path in (profile_path, CALIBRATION_TRACE_PATH, PREDICTION_TRACE_PATH, BUSINESS_REPORT_PATH, METADATA_PATH, README_PATH):
        if path.exists():
            path.unlink()
    reset_session(CASE_NO)
    rng = np.random.default_rng(SYNTHETIC_SEED)

    bridge = EEGAlgorithmBridge(
        case_no=CASE_NO,
        age=AGE,
        gender=GENDER,
        sample_rate=SAMPLE_RATE,
        profile_dir=ARTIFACT_DIR,
        runtime_config=ALGORITHM_RUNTIME_CONFIG,
    )

    _begin_clean_trace_file(CALIBRATION_TRACE_PATH, _trace_header())
    calibration_result, _ = _run_calibration(bridge, rng)
    if not profile_path.exists():
        raise RuntimeError(f"profile was not created: {profile_path}")

    _begin_clean_trace_file(
        PREDICTION_TRACE_PATH,
        _trace_header(profile_source="SYNTHETIC CALIBRATION FROM SAME RUN."),
    )
    prediction_windows, _ = _run_prediction(bridge, rng)

    profile_payload = _load_profile_payload(profile_path)
    _write_business_report(
        profile_payload=profile_payload,
        prediction_windows=prediction_windows,
    )
    _write_metadata(
        profile_path=profile_path,
        calibration_result=calibration_result,
        prediction_windows=prediction_windows,
    )
    _write_readme(
        profile_path=profile_path,
        calibration_result=calibration_result,
        prediction_windows=prediction_windows,
    )

    print("Synthetic EEG trace run completed.")
    print(f"Profile: {profile_path}")
    print(f"Calibration trace: {CALIBRATION_TRACE_PATH}")
    print(f"Prediction trace: {PREDICTION_TRACE_PATH}")
    print(f"Business report: {BUSINESS_REPORT_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"README: {README_PATH}")
    print(f"Accepted/rejected/attempts: {calibration_result.get('accepted_trials')}/"
          f"{calibration_result.get('rejected_trials')}/{calibration_result.get('attempts')}")
    print(f"T_all: {profile_payload['action_threshold']}")
    print(f"Prediction windows: {len(prediction_windows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
