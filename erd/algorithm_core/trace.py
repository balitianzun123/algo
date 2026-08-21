"""Human-readable EEG trace formatting helpers."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import constants

logger = logging.getLogger(__name__)

TRACE_LOG_PATH = Path("logs") / "eeg_trace_observation.log"
_FILE_HANDLER_READY = False
_SESSION_KEYS: set[tuple[object, ...]] = set()


def _trace_enabled() -> bool:
    return bool(getattr(constants, "EEG_TRACE_ENABLED", False))


def _trace_detail_enabled() -> bool:
    return _trace_enabled() and bool(getattr(constants, "EEG_TRACE_DETAIL", False))


def _fmt(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return str(number)
    return f"{number:.{digits}f}"


def _fmt_signed(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return str(number)
    return f"{number:+.{digits}f}"


def _ensure_file_handler(target: logging.Logger) -> None:
    global _FILE_HANDLER_READY
    if _FILE_HANDLER_READY or target is not logger:
        return
    try:
        TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(TRACE_LOG_PATH, encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        target.addHandler(handler)
        target.setLevel(logging.INFO)
        _FILE_HANDLER_READY = True
    except Exception:
        _FILE_HANDLER_READY = True


def trace_log_path() -> Path:
    return TRACE_LOG_PATH


def trace_block(
    module: str,
    lines: Sequence[object],
    *,
    detail: bool = False,
    logger_: logging.Logger | None = None,
) -> None:
    if detail:
        if not _trace_detail_enabled():
            return
    elif not _trace_enabled():
        return
    target = logger_ or logger
    _ensure_file_handler(target)
    if not target.isEnabledFor(logging.INFO):
        return
    body = "\n".join(str(line) for line in lines)
    prefix = "EEG_TRACE_DETAIL" if detail else "EEG_TRACE"
    try:
        target.info("[%s][%s]\n%s", prefix, module, body)
    except Exception:
        return


def trace_line(
    module: str,
    line: object,
    *,
    detail: bool = False,
    logger_: logging.Logger | None = None,
) -> None:
    if detail:
        if not _trace_detail_enabled():
            return
    elif not _trace_enabled():
        return
    target = logger_ or logger
    _ensure_file_handler(target)
    if not target.isEnabledFor(logging.INFO):
        return
    prefix = "EEG_TRACE_DETAIL" if detail else "EEG_TRACE"
    try:
        target.info("[%s][%s] %s", prefix, module, line)
    except Exception:
        return


def reset_session_trace(case_no: str | None = None) -> None:
    if case_no is None:
        _SESSION_KEYS.clear()
        return
    case = str(case_no)
    stale = {key for key in _SESSION_KEYS if len(key) > 1 and key[1] == case}
    _SESSION_KEYS.difference_update(stale)


def _array(values: object) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _range(values: object, digits: int = 2) -> str:
    array = _array(values)
    if not array.size or not np.isfinite(array).all():
        return "n/a"
    return f"{_fmt(np.min(array), digits)}~{_fmt(np.max(array), digits)}"


def _rms_by_channel(data: object) -> np.ndarray:
    array = _array(data)
    if array.ndim != 2:
        return np.asarray([], dtype=np.float64)
    return np.sqrt(np.mean(np.square(array), axis=1))


def build_preprocess_summary(
    *,
    raw_selected: object,
    filtered_epoch: object,
    sampling_rate: float,
    channel_names: Sequence[str],
) -> dict[str, object]:
    raw = _array(raw_selected)
    epoch = _array(filtered_epoch)
    raw_rms = _rms_by_channel(raw)
    filtered_rms = _rms_by_channel(epoch)
    return {
        "input_shape": tuple(int(value) for value in raw.shape),
        "output_shape": tuple(int(value) for value in epoch.shape),
        "sample_rate": float(sampling_rate),
        "channel_names": tuple(str(channel) for channel in channel_names),
        "raw_rms": [float(value) for value in raw_rms],
        "filtered_rms": [float(value) for value in filtered_rms],
        "raw_rms_range": _range(raw_rms),
        "filtered_rms_range": _range(filtered_rms),
        "finite": bool(np.isfinite(epoch).all()) if epoch.size else False,
        "all_zero": bool(epoch.size and np.all(epoch == 0.0)),
        "shape_ok": bool(epoch.ndim == 2 and epoch.shape[0] == len(tuple(channel_names))),
    }


def trace_session(
    *,
    case_no: str,
    mode: str,
    sample_rate: float,
    input_channels: int,
    required_channels: Sequence[str],
    action_channels: Sequence[str],
    window_seconds: float,
    step_seconds: float | None = None,
    mapping: Mapping[str, int] | None = None,
    status: str = "READY",
    force: bool = False,
    logger_: logging.Logger | None = None,
) -> None:
    mapping_key = tuple(sorted((str(key), int(value)) for key, value in dict(mapping or {}).items()))
    key = (
        str(mode),
        str(case_no),
        float(sample_rate),
        int(input_channels),
        tuple(str(channel) for channel in required_channels),
        tuple(str(channel) for channel in action_channels),
        float(window_seconds),
        None if step_seconds is None else float(step_seconds),
        mapping_key,
        str(status),
    )
    if not force and key in _SESSION_KEYS:
        return
    _SESSION_KEYS.add(key)
    step_text = "n/a" if step_seconds is None else f"{_fmt(step_seconds, 1)} s"
    lines = [
        f"Case        : {case_no}",
        f"Mode        : {mode}",
        f"Sample Rate : {_fmt(sample_rate, 1)} Hz",
        f"Input       : {int(input_channels)} channels",
        f"Algorithm   : {' '.join(str(channel) for channel in required_channels)}",
        f"Action ROI  : {' '.join(str(channel) for channel in action_channels)}",
        f"Window      : {_fmt(window_seconds, 1)} s",
        f"Step        : {step_text}",
        f"Status      : {status}",
    ]
    trace_block("SESSION", lines, logger_=logger_)


def trace_adc_to_uv(
    raw_counts: object,
    corrected_uv: object | None,
    *,
    raw_adc_offset: float,
    raw_adc_scale: float,
    raw_uv: object | None = None,
    logger_: logging.Logger | None = None,
) -> None:
    raw = _array(raw_counts)
    corrected = None if corrected_uv is None else _array(corrected_uv)
    channel_count = int(raw.size) if raw.size else 16
    raw_ok = bool(raw.size and np.isfinite(raw).all())
    corrected_ok = bool(
        corrected is not None
        and corrected.size
        and np.isfinite(corrected).all()
    )
    if raw_ok and corrected_ok:
        line = (
            f"{channel_count}ch | ADC {_fmt(np.min(raw), 0)}~{_fmt(np.max(raw), 0)} | "
            f"uV {_fmt_signed(np.min(corrected))}~{_fmt_signed(np.max(corrected))} | "
            f"offset={_fmt(raw_adc_offset, 0)} | scale={_fmt(raw_adc_scale, 7)} | status=OK"
        )
    elif raw_ok:
        uv = raw if raw_uv is None else _array(raw_uv)
        line = (
            f"{channel_count}ch | ADC {_fmt(np.min(raw), 0)}~{_fmt(np.max(raw), 0)} | "
            f"uV {_fmt_signed(np.min(uv))}~{_fmt_signed(np.max(uv))} | "
            f"offset={_fmt(raw_adc_offset, 0)} | scale={_fmt(raw_adc_scale, 7)} | status=OK"
        )
    else:
        line = (
            f"{channel_count}ch | ADC finite={'yes' if raw_ok else 'no'} | "
            f"uV={'finite' if corrected_ok else 'n/a'} | status=CHECK"
        )
    trace_block("ADC", [line], logger_=logger_)
    raw_uv_array = None if raw_uv is None else _array(raw_uv)
    trace_block(
        "ADC",
        [
            f"raw_counts={raw.tolist()}",
            f"raw_uV={'n/a' if raw_uv_array is None else raw_uv_array.tolist()}",
            f"uV={'n/a' if corrected is None else corrected.tolist()}",
        ],
        detail=True,
        logger_=logger_,
    )


def trace_channel_map(
    *,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    selected_channels: Sequence[str],
    missing_channels: Sequence[str],
    sampling_rate: float | None = None,
    logger_: logging.Logger | None = None,
) -> None:
    lines = [
        f"input_shape={list(input_shape)}",
        f"selected={' '.join(str(channel) for channel in selected_channels)}",
        f"output_shape={list(output_shape)}",
        f"sample_rate={sampling_rate if sampling_rate is not None else 'n/a'}",
        f"missing={'none' if not missing_channels else ' '.join(str(channel) for channel in missing_channels)}",
    ]
    trace_block("CHANNEL_MAP", lines, detail=True, logger_=logger_)


def trace_preprocess(
    *,
    trial_id: str,
    sampling_rate: float,
    raw_selected: object,
    filtered_epoch: object,
    channel_names: Sequence[str],
    notch_hz: float,
    bandpass_hz: Sequence[float],
    logger_: logging.Logger | None = None,
) -> None:
    summary = build_preprocess_summary(
        raw_selected=raw_selected,
        filtered_epoch=filtered_epoch,
        sampling_rate=sampling_rate,
        channel_names=channel_names,
    )
    lines = [
        f"trial_id={trial_id}",
        f"input_shape={summary['input_shape']}",
        f"output_shape={summary['output_shape']}",
        f"process={_fmt(notch_hz, 1)} Hz notch + {_fmt(bandpass_hz[0], 1)}~{_fmt(bandpass_hz[1], 1)} Hz bandpass",
        f"raw_rms_range={summary['raw_rms_range']} uV",
        f"filtered_rms_range={summary['filtered_rms_range']} uV",
        f"finite={'yes' if summary['finite'] else 'no'}",
        f"all_zero={'yes' if summary['all_zero'] else 'no'}",
        "",
        "channel raw_rms filtered_rms",
    ]
    for channel, raw_rms, filtered_rms in zip(
        summary["channel_names"],
        summary["raw_rms"],
        summary["filtered_rms"],
    ):
        lines.append(f"{channel} {_fmt(raw_rms)} {_fmt(filtered_rms)}")
    trace_block("PREPROCESS", lines, detail=True, logger_=logger_)


def _status_by_channel(quality_result: Mapping[str, object] | None) -> dict[str, str]:
    channels = [] if quality_result is None else quality_result.get("channels", [])
    status: dict[str, str] = {}
    if isinstance(channels, (list, tuple)):
        for item in channels:
            if isinstance(item, Mapping):
                name = str(item.get("name", ""))
                if name:
                    status[name] = str(item.get("status", "unknown")).upper()
    return status


def quality_summary(
    *,
    quality_result: Mapping[str, object] | None,
    action_channels: Sequence[str],
    bad_channels: Sequence[str],
) -> dict[str, object]:
    status = _status_by_channel(quality_result)
    good_count = len(tuple(action_channels)) - len(tuple(bad_channels))
    return {
        "statuses": {channel: status.get(channel, "UNKNOWN") for channel in action_channels},
        "good_count": int(good_count),
        "total_count": len(tuple(action_channels)),
        "bad_channels": tuple(str(channel) for channel in bad_channels),
    }


def trace_quality(
    *,
    quality_result: Mapping[str, object] | None,
    action_channels: Sequence[str],
    bad_channels: Sequence[str],
    quality_mode: str,
    threshold_source: str | None = None,
    used_channels: Sequence[str] | None = None,
    logger_: logging.Logger | None = None,
) -> None:
    summary = quality_summary(
        quality_result=quality_result,
        action_channels=action_channels,
        bad_channels=bad_channels,
    )
    lines = [
        f"mode={quality_mode}",
        f"bad={'none' if not bad_channels else ' '.join(str(channel) for channel in bad_channels)}",
        f"used={' '.join(str(channel) for channel in used_channels or ())}",
        f"threshold_source={threshold_source or 'n/a'}",
        f"statuses={summary['statuses']}",
    ]
    trace_block("QUALITY", lines, detail=True, logger_=logger_)


def trace_power(
    *,
    channel_powers: Mapping[str, float],
    used_channels: Sequence[str],
    action_channels: Sequence[str],
    band_hz: Sequence[float],
    window_s: Sequence[float],
    logger_: logging.Logger | None = None,
) -> None:
    lines = [
        f"band={_fmt(band_hz[0], 1)}~{_fmt(band_hz[1], 1)} Hz",
        f"window={_fmt(window_s[0], 1)}~{_fmt(window_s[1], 1)} s",
        f"used={' '.join(str(channel) for channel in used_channels)}",
        f"powers={dict(channel_powers)}",
        f"action_channels={' '.join(str(channel) for channel in action_channels)}",
    ]
    trace_block("POWER", lines, detail=True, logger_=logger_)


def trace_erd(
    *,
    channel_powers: Mapping[str, float],
    channel_baselines: Mapping[str, float],
    channel_erd: Mapping[str, float],
    channels: Sequence[str],
    baseline_type: str,
    holdout_trial: str | None = None,
    logger_: logging.Logger | None = None,
) -> None:
    lines = [
        f"baseline_type={baseline_type}",
        f"holdout_trial={holdout_trial or 'n/a'}",
        f"channels={' '.join(str(channel) for channel in channels)}",
        f"powers={dict(channel_powers)}",
        f"baselines={dict(channel_baselines)}",
        f"erd={dict(channel_erd)}",
    ]
    trace_block("ERD", lines, detail=True, logger_=logger_)


def trace_roi_score(
    details: Mapping[str, Any],
    *,
    logger_: logging.Logger | None = None,
) -> None:
    trace_block("ROI_SCORE", [repr(dict(details))], detail=True, logger_=logger_)


def _term_by_channel(roi_details: Mapping[str, Any] | None) -> dict[str, dict[str, float]]:
    terms: dict[str, dict[str, float]] = {}
    if not roi_details:
        return terms
    for side in ("left", "right"):
        roi = roi_details.get(side, {})
        if not isinstance(roi, Mapping):
            continue
        for term in roi.get("terms", []):
            if isinstance(term, Mapping) and "channel" in term:
                channel = str(term["channel"])
                terms[channel] = {
                    "weight": float(term.get("weight", float("nan"))),
                    "weighted": float(term.get("weighted", float("nan"))),
                }
    return terms


def _roi_line(label: str, roi: Mapping[str, Any] | None) -> str:
    if not roi:
        return f"{label:<10}: n/a"
    weighted_sum = float(roi.get("weighted_sum", float("nan")))
    weight_sum = float(roi.get("weight_sum", float("nan")))
    score = float(roi.get("score", float("nan")))
    return f"{label:<10}: -({_fmt_signed(weighted_sum)}) / {_fmt(weight_sum)} = {_fmt(score)}"


def _channel_table_line(
    channel: str,
    *,
    used: set[str],
    powers: Mapping[str, float],
    baselines: Mapping[str, float],
    erd: Mapping[str, float],
    terms: Mapping[str, Mapping[str, float]],
) -> str:
    if channel not in used:
        return f"{channel:<8} EXCLUDED"
    term = terms.get(channel, {})
    return (
        f"{channel:<8} "
        f"{_fmt(powers.get(channel, float('nan'))):>9} "
        f"{_fmt(baselines.get(channel, float('nan'))):>12} "
        f"{_fmt_signed(erd.get(channel, float('nan'))):>10}% "
        f"{_fmt(term.get('weight', float('nan')), 1):>5} "
        f"{_fmt_signed(term.get('weighted', float('nan'))):>14}"
    )


def _display_channels(channels: Sequence[str]) -> tuple[str, ...]:
    selected = {str(channel) for channel in channels}
    ordered = [channel for channel in ("C3", "CP3", "FC3", "C4", "CP4", "FC4") if channel in selected]
    ordered.extend(str(channel) for channel in channels if str(channel) not in ordered)
    return tuple(ordered)


def _format_trial_values(trial_ids: Sequence[str], values: Sequence[float], *, digits: int = 6) -> str:
    return " | ".join(
        f"{trial_id}={_fmt(value, digits)}"
        for trial_id, value in zip(trial_ids, values)
    )


def _middle_values(sorted_values: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in sorted_values)
    if not values:
        return tuple()
    middle = len(values) // 2
    if len(values) % 2:
        return (values[middle],)
    return (values[middle - 1], values[middle])


def _cal_power_lines(channel_powers: Mapping[str, float], channels: Sequence[str]) -> list[str]:
    lines = ["", "Rest Power", "Left ROI"]
    for channel in ("C3", "CP3", "FC3"):
        if channel in channels and channel in channel_powers:
            lines.append(f"{channel:<4}= {_fmt(channel_powers[channel], 6)}")
    lines.append("")
    lines.append("Right ROI")
    for channel in ("C4", "CP4", "FC4"):
        if channel in channels and channel in channel_powers:
            lines.append(f"{channel:<4}= {_fmt(channel_powers[channel], 6)}")
    return lines


def trace_calibration_baseline_derivation(
    *,
    rest_power_details: Mapping[str, Any],
    criterion_details: Mapping[str, Any],
    thresholds_without_channel: Mapping[str, float],
    logger_: logging.Logger | None = None,
) -> None:
    """Emit the formal Rest Power baseline calibration ledger."""
    trace_rest_power_matrix(rest_power_details, logger_=logger_)
    trace_final_baseline_derivation(rest_power_details, logger_=logger_)
    trace_criterion_configuration(
        criterion_details=criterion_details,
        thresholds_without_channel=thresholds_without_channel,
        logger_=logger_,
    )


def trace_criterion_configuration(
    *,
    criterion_details: Mapping[str, Any],
    thresholds_without_channel: Mapping[str, float],
    logger_: logging.Logger | None = None,
) -> None:
    lines = [
        "=" * 60,
        "MI DECISION CRITERION CONFIGURATION",
        "=" * 60,
        f"Function : {criterion_details.get('criterion_function', 'get_mi_success_drop_threshold_pct')}",
        f"Name     : {criterion_details.get('criterion_name', 'fixed_protocol_mi_success_drop_pct')}",
        f"Version  : {criterion_details.get('criterion_version', 'n/a')}",
        "Meaning  : selected ROI Power Drop must reach this percent.",
        "Source   : fixed protocol criterion, not learned from Rest ERD.",
        "",
        f"T_all    = {_fmt(criterion_details.get('T_all', criterion_details.get('threshold_pct', float('nan'))), 6)}%",
        "",
        "Single-channel thresholds",
        "-------------------------",
    ]
    for channel in constants.ACTION_CHANNELS:
        if channel in thresholds_without_channel:
            lines.append(f"T_no_{channel:<3}= {_fmt(thresholds_without_channel[channel], 6)}%")
    lines.append("=" * 60)
    trace_block("CALIBRATION][CRITERION", lines, logger_=logger_)


def trace_rest_power_matrix(
    rest_power_details: Mapping[str, Any],
    *,
    logger_: logging.Logger | None = None,
) -> None:
    trial_ids = tuple(str(trial_id) for trial_id in rest_power_details.get("trial_ids", ()))
    rows = [dict(row) for row in rest_power_details.get("power_rows", [])]
    channels = _display_channels(rest_power_details.get("channels", constants.ACTION_CHANNELS))
    labels = tuple(f"R{index:02d}" for index in range(1, len(rows) + 1))
    lines = [
        "=" * 60,
        "REST POWER MATRIX",
        "=" * 60,
        f"{'Trial':<8}" + "".join(f"{channel:>10}" for channel in channels),
        "-" * (8 + 10 * len(channels)),
    ]
    for label, row in zip(labels, rows):
        lines.append(f"{label:<8}" + "".join(f"{_fmt(row.get(channel, float('nan')), 4):>10}" for channel in channels))
    lines.append("=" * 60)
    trace_block("CALIBRATION][REST_POWER_MATRIX", lines, logger_=logger_)
    if _trace_detail_enabled() and trial_ids:
        detail_lines = ["Short label -> real trial_id"]
        detail_lines.extend(f"{label} -> {trial_id}" for label, trial_id in zip(labels, trial_ids))
        trace_block("CALIBRATION][REST_POWER_MATRIX][TRIAL_IDS", detail_lines, detail=True, logger_=logger_)


def trace_final_baseline_derivation(
    baseline_details: Mapping[str, Any],
    *,
    logger_: logging.Logger | None = None,
) -> None:
    trial_ids = tuple(str(trial_id) for trial_id in baseline_details.get("trial_ids", ()))
    channels = _display_channels(baseline_details.get("channels", constants.ACTION_CHANNELS))
    channel_details = dict(baseline_details.get("channel_details", {}))
    baselines = dict(baseline_details.get("baselines", {}))
    rule = str(baseline_details.get("aggregation", "median"))
    lines = [
        "=" * 60,
        f"Rule : {rule} of R01~R{len(trial_ids):02d} Rest Power",
        "=" * 60,
        "",
        "Channel   RawN CleanN Outliers   Clean Middle Values          Final Baseline",
        "-" * 72,
    ]
    for channel in channels:
        details = dict(channel_details.get(channel, {}))
        source_values = list(details.get("source_values", []))
        clean_sorted_values = list(details.get("clean_sorted_values", details.get("sorted_values", [])))
        middle_values = _middle_values(clean_sorted_values)
        iqr_filter = dict(details.get("iqr_filter", {}))
        outlier_count = len(list(iqr_filter.get("outliers", [])))
        clean_count = int(details.get("clean_power_count", len(clean_sorted_values)))
        if channel == "C4":
            lines.append("")
        lines.append(
            f"{channel:<8}"
            f"{len(source_values):>4}"
            f"{clean_count:>7}"
            f"{outlier_count:>9}   "
            f"{' | '.join(_fmt(value, 6) for value in middle_values):<30}"
            f"{_fmt(baselines.get(channel, float('nan')), 6):>16}"
        )
    lines.append("=" * 60)
    trace_block("CALIBRATION][FINAL_BASELINE", lines, logger_=logger_)
    if not _trace_detail_enabled():
        return
    detail_lines = [
        "=" * 60,
        f"Source Trials: {' '.join(trial_ids)}",
        "",
    ]
    for channel in channels:
        details = dict(channel_details.get(channel, {}))
        values = list(details.get("source_values", []))
        sorted_values = list(details.get("sorted_values", []))
        clean_values = list(details.get("clean_values", values))
        clean_sorted_values = list(details.get("clean_sorted_values", sorted_values))
        iqr_filter = dict(details.get("iqr_filter", {}))
        outliers = list(iqr_filter.get("outliers", []))
        outlier_text = "none"
        if outliers:
            outlier_text = " | ".join(
                f"{item.get('trial_id', 'unknown')}={_fmt(item.get('power', float('nan')), 6)} {item.get('reason', 'outlier')}"
                for item in outliers
                if isinstance(item, Mapping)
            )
        detail_lines.extend(
            [
                f"{channel}:",
                "Raw Rest Powers:",
                _format_trial_values(tuple(f"R{index:02d}" for index in range(1, len(values) + 1)), values, digits=6),
                f"Current baseline rule: {rule}",
                "Sorted Raw Powers:",
                " | ".join(_fmt(value, 6) for value in sorted_values),
                f"Q1          = {_fmt(iqr_filter.get('q1', float('nan')), 6)}",
                f"Q3          = {_fmt(iqr_filter.get('q3', float('nan')), 6)}",
                f"IQR         = {_fmt(iqr_filter.get('iqr', float('nan')), 6)}",
                f"Lower Fence = {_fmt(iqr_filter.get('lower_fence', float('nan')), 6)}",
                f"Upper Fence = {_fmt(iqr_filter.get('upper_fence', float('nan')), 6)}",
                f"IQR Outliers: {outlier_text}",
                "Clean Rest Powers:",
                " | ".join(_fmt(value, 6) for value in clean_values),
                "Clean Sorted Powers:",
                " | ".join(_fmt(value, 6) for value in clean_sorted_values),
                f"Final {channel} Baseline = {_fmt(baselines.get(channel, float('nan')), 12)}",
                "",
            ]
        )
    detail_lines.append("=" * 60)
    trace_block("CALIBRATION][FINAL_BASELINE", detail_lines, detail=True, logger_=logger_)


def trace_prediction_window_summary(
    *,
    backend_trial_count: int | None = None,
    trial_prediction_index: int | None = None,
    prediction_index: int,
    window_start_s: float,
    window_end_s: float,
    sample_rate: float,
    channel_count: int,
    sample_count: int,
    quality_mode: str,
    used_channels: Sequence[str],
    bad_channels: Sequence[str],
    channel_powers: Mapping[str, float] | None = None,
    channel_baselines: Mapping[str, float] | None = None,
    channel_erd: Mapping[str, float] | None = None,
    roi_details: Mapping[str, Any] | None,
    threshold_source: str,
    action_score: float | None,
    action_threshold: float | None,
    score_margin: float | None,
    preprocess_summary: Mapping[str, object] | None = None,
    code: int | None = None,
    logger_: logging.Logger | None = None,
) -> None:
    powers = dict(channel_powers or {})
    baselines = dict(channel_baselines or {})
    erd = dict(channel_erd or {})
    terms = _term_by_channel(roi_details)
    used = {str(channel) for channel in used_channels}
    pre = dict(preprocess_summary or {})
    finite_ok = pre.get("finite", True)
    all_zero = pre.get("all_zero", False)
    preprocess_status = "OK" if finite_ok and not all_zero else "CHECK"
    rms_text = (
        f"DC-corrected {pre.get('raw_rms_range', 'n/a')} uV "
        f"-> filtered {pre.get('filtered_rms_range', 'n/a')} uV"
    )
    left = roi_details.get("left", {}) if isinstance(roi_details, Mapping) else {}
    right = roi_details.get("right", {}) if isinstance(roi_details, Mapping) else {}
    backend_trial_text = "N/A" if backend_trial_count is None else str(int(backend_trial_count))
    trial_window_text = "N/A" if trial_prediction_index is None else str(int(trial_prediction_index))
    lines = [
        "=" * 60,
        (
            f"TRIAL #{backend_trial_text} | TRIAL WINDOW #{trial_window_text} | "
            f"GLOBAL WINDOW #{prediction_index} | {_fmt(window_start_s, 1)}s -> {_fmt(window_end_s, 1)}s"
        ),
        "=" * 60,
        f"Data       : {channel_count}ch x {sample_count} | {_fmt(sample_rate, 1)} Hz | {_fmt(window_end_s - window_start_s, 1)} s",
        (
            f"Preprocess : {_fmt(constants.PREPROCESSING_NOTCH_HZ, 1)}Hz notch + "
            f"{_fmt(constants.PREPROCESSING_BANDPASS_HZ[0], 1)}~{_fmt(constants.PREPROCESSING_BANDPASS_HZ[1], 1)}Hz bandpass | {preprocess_status}"
        ),
        f"RMS        : {rms_text}",
        f"Quality    : {len(used)}/6 | bad={'none' if not bad_channels else ' '.join(str(channel) for channel in bad_channels)}",
        f"Mode       : {quality_mode}",
        "",
        "Channel   Power      Baseline        ERD     W   Contribution",
        "-" * 64,
    ]
    for channel in ("C3", "CP3", "FC3"):
        lines.append(
            _channel_table_line(
                channel,
                used=used,
                powers=powers,
                baselines=baselines,
                erd=erd,
                terms=terms,
            )
        )
    lines.append("")
    for channel in ("C4", "CP4", "FC4"):
        lines.append(
            _channel_table_line(
                channel,
                used=used,
                powers=powers,
                baselines=baselines,
                erd=erd,
                terms=terms,
            )
        )
    lines.extend(
        [
            "-" * 64,
            "",
            _roi_line("Left ROI", left),
            _roi_line("Right ROI", right),
            "",
            f"Action Score : max({_fmt(left.get('score', float('nan')))}, {_fmt(right.get('score', float('nan')))}) = {_fmt(action_score) if action_score is not None else 'n/a'}",
            (
                f"Threshold    : {threshold_source} = {_fmt(action_threshold)}"
                if action_threshold is not None
                else f"Threshold    : {threshold_source}"
            ),
            f"Margin       : {_fmt_signed(score_margin) if score_margin is not None else 'n/a'}",
        ]
    )
    if code is not None:
        lines.append(f"Code         : {code}")
    lines.append("=" * 60)
    trace_block(f"WINDOW#{prediction_index}", lines, logger_=logger_)


def trace_calibration_trial(
    *,
    attempt: int,
    accepted: bool,
    quality_result: Mapping[str, object] | None,
    action_channels: Sequence[str],
    bad_channels: Sequence[str],
    accepted_count: int,
    required_count: int,
    trial_start_s: float | None = None,
    trial_end_s: float | None = None,
    trial_powers: Mapping[str, float] | None = None,
    trial_power_error: str | None = None,
    logger_: logging.Logger | None = None,
) -> None:
    summary = quality_summary(
        quality_result=quality_result,
        action_channels=action_channels,
        bad_channels=bad_channels,
    )
    status = "ACCEPT" if accepted else "REJECT"
    bad = "none" if not bad_channels else " ".join(str(channel) for channel in bad_channels)
    time_text = "n/a" if trial_start_s is None or trial_end_s is None else f"{_fmt(trial_start_s, 1)}s->{_fmt(trial_end_s, 1)}s"
    line = (
        f"{time_text} | {status} | quality={summary['good_count']}/{summary['total_count']} | "
        f"bad={bad} | accepted={accepted_count}/{required_count}"
    )
    trace_line(f"CAL#{attempt:02d}", line, logger_=logger_)
    if trial_powers:
        lines = [
            f"attempt={attempt:02d}",
            f"time={time_text}",
            f"decision={status}",
        ]
        lines.extend(_cal_power_lines(trial_powers, action_channels))
        trace_block(f"CAL#{attempt:02d}][REST_POWER", lines, detail=True, logger_=logger_)
    elif trial_power_error:
        trace_block(
            f"CAL#{attempt:02d}][REST_POWER",
            [f"Rest Power : unavailable | {trial_power_error}"],
            detail=True,
            logger_=logger_,
        )


def trace_calibration_summary(
    *,
    accepted_trials: int,
    rejected_trials: int,
    attempts: int,
    action_threshold: float,
    thresholds_without_channel: Mapping[str, float],
    baselines: Mapping[str, float],
    threshold_source: str = "mi_success_drop_pct",
    logger_: logging.Logger | None = None,
) -> None:
    lines = [
        "=" * 50,
        "REST POWER BASELINE CALIBRATION COMPLETE",
        "=" * 50,
        f"Accepted : {accepted_trials}",
        f"Rejected : {rejected_trials if rejected_trials >= 0 else 'n/a'}",
        f"Attempts : {attempts if attempts >= 0 else 'n/a'}",
        "",
        "Baseline",
        "Rule : per-channel one-pass Tukey 1.5 IQR filter, then median(Clean Rest Power)",
    ]
    for channel in ("FC3", "C3", "CP3", "", "FC4", "C4", "CP4"):
        if not channel:
            lines.append("")
        elif channel in baselines:
            lines.append(f"{channel:<4}= {_fmt(baselines[channel])}")
    lines.extend(
        [
            "",
            "MI Decision Criterion",
            f"Source : {threshold_source}",
            f"T_all  = {_fmt(action_threshold)}%",
            "",
            "Single-channel Thresholds",
        ]
    )
    for channel in constants.ACTION_CHANNELS:
        if channel in thresholds_without_channel:
            lines.append(f"T_no_{channel:<3}= {_fmt(thresholds_without_channel[channel])}%")
    lines.append("=" * 50)
    trace_block("CALIBRATION", lines, logger_=logger_)
