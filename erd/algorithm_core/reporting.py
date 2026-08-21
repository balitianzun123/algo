"""Report-only helpers for EEG calibration and prediction audit text."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


LEFT_REPORT_CHANNELS = ("C3", "CP3", "FC3")
RIGHT_REPORT_CHANNELS = ("C4", "CP4", "FC4")
BASELINE_REPORT_CHANNELS = ("C3", "CP3", "FC3", "C4", "CP4", "FC4")


def _as_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: object, digits: int = 4, *, signed: bool = False) -> str:
    number = _as_float(value)
    prefix = "+" if signed else ""
    return f"{number:{prefix}.{digits}f}"


def _fmt_optional(value: object) -> str:
    return "N/A" if value is None else str(value)


def _terms_by_channel(roi_details: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    terms: dict[str, Mapping[str, Any]] = {}
    for side in ("left", "right"):
        roi = roi_details.get(side, {})
        if not isinstance(roi, Mapping):
            continue
        for term in roi.get("terms", ()):
            if isinstance(term, Mapping) and "channel" in term:
                terms[str(term["channel"])] = term
    return terms


def build_action_audit_summary(
    *,
    powers: Mapping[str, float],
    baselines: Mapping[str, float],
    channel_erd: Mapping[str, float],
    roi_details: Mapping[str, Any],
    channels: Sequence[str],
    action_score: float,
    index: int | None = None,
    label: str | None = None,
    trial_id: str | None = None,
    baseline_type: str | None = None,
) -> dict[str, Any]:
    """Collect already-computed Power/Baseline/ERD/ROI values for reports."""
    selected = tuple(str(channel) for channel in channels)
    terms = _terms_by_channel(roi_details)
    channel_summary: dict[str, dict[str, float]] = {}
    for channel in selected:
        term = terms.get(channel)
        if term is None:
            continue
        channel_summary[channel] = {
            "power": float(powers[channel]),
            "baseline": float(baselines[channel]),
            "erd": float(channel_erd[channel]),
            "power_drop": float(-float(channel_erd[channel])),
            "weight": float(term.get("weight", 0.0)),
            "contribution": float(term.get("weighted", term.get("contribution", 0.0))),
        }

    left = dict(roi_details.get("left", {})) if isinstance(roi_details, Mapping) else {}
    right = dict(roi_details.get("right", {})) if isinstance(roi_details, Mapping) else {}
    summary: dict[str, Any] = {
        "channels": channel_summary,
        "used_channels": selected,
        "left_weighted_sum": float(left.get("weighted_sum", 0.0)),
        "left_weight_sum": float(left.get("weight_sum", 0.0)),
        "left_score": float(left.get("score", 0.0)),
        "left_roi_erd_pct": float(left.get("erd_pct", -float(left.get("score", 0.0)))),
        "left_roi_drop_pct": float(left.get("drop_pct", left.get("score", 0.0))),
        "right_weighted_sum": float(right.get("weighted_sum", 0.0)),
        "right_weight_sum": float(right.get("weight_sum", 0.0)),
        "right_score": float(right.get("score", 0.0)),
        "right_roi_erd_pct": float(right.get("erd_pct", -float(right.get("score", 0.0)))),
        "right_roi_drop_pct": float(right.get("drop_pct", right.get("score", 0.0))),
        "selected_roi": str(roi_details.get("selected_roi", "left") if isinstance(roi_details, Mapping) else "left"),
        "selected_roi_drop_pct": float(roi_details.get("selected_roi_drop_pct", action_score) if isinstance(roi_details, Mapping) else action_score),
        "action_score": float(action_score),
    }
    if index is not None:
        summary["index"] = int(index)
    if label is not None:
        summary["label"] = str(label)
    if trial_id is not None:
        summary["trial_id"] = str(trial_id)
    if baseline_type is not None:
        summary["baseline_type"] = str(baseline_type)
    return summary


def build_prediction_report_summary(
    *,
    powers: Mapping[str, float],
    baselines: Mapping[str, float],
    channel_erd: Mapping[str, float],
    roi_details: Mapping[str, Any],
    used_channels: Sequence[str],
    bad_channels: Sequence[str],
    quality_mode: str,
    threshold_source: str,
    action_score: float,
    action_threshold: float,
    score_margin: float,
    backend_trial_count: int | None,
    trial_prediction_index: int | None,
    prediction_index: int,
    algorithm_window_result: str,
) -> dict[str, Any]:
    summary = build_action_audit_summary(
        powers=powers,
        baselines=baselines,
        channel_erd=channel_erd,
        roi_details=roi_details,
        channels=used_channels,
        action_score=action_score,
        label=f"WINDOW#{prediction_index}",
        baseline_type="personal_profile",
    )
    bad_text = "none" if not bad_channels else " ".join(str(channel) for channel in bad_channels)
    criterion_met = float(action_score) >= float(action_threshold)
    selected_roi = str(summary.get("selected_roi", "left"))
    selected_drop = float(summary.get("selected_roi_drop_pct", action_score))
    decision_detail = {
        "left_roi_erd_pct": float(summary.get("left_roi_erd_pct", 0.0)),
        "right_roi_erd_pct": float(summary.get("right_roi_erd_pct", 0.0)),
        "left_roi_drop_pct": float(summary.get("left_roi_drop_pct", 0.0)),
        "right_roi_drop_pct": float(summary.get("right_roi_drop_pct", 0.0)),
        "selected_roi": selected_roi,
        "selected_roi_drop_pct": selected_drop,
        "threshold_pct": float(action_threshold),
        "threshold_source": str(threshold_source),
        "comparison_operator": ">=",
        "comparison_expression": f"{selected_drop:.4f} >= {float(action_threshold):.4f}",
        "criterion_met": bool(criterion_met),
        "decision_text": (
            f"{selected_roi} ROI Power Drop {selected_drop:.4f}% "
            f"{'reaches' if criterion_met else 'does not reach'} "
            f"MI threshold {float(action_threshold):.4f}%, therefore code={1 if criterion_met else 0}"
        ),
    }
    if bad_channels:
        decision_detail["excluded_channel"] = str(tuple(bad_channels)[0])
    summary.update(
        {
            "quality_text": f"{len(tuple(used_channels))}/6 | bad={bad_text}",
            "quality_mode": str(quality_mode),
            "bad_channels": tuple(str(channel) for channel in bad_channels),
            "threshold_source": str(threshold_source),
            "action_threshold": float(action_threshold),
            "score_margin": float(score_margin),
            "algorithm_window_result": str(algorithm_window_result),
            "decision_detail": decision_detail,
            "window_identity": {
                "backend_trial_count": backend_trial_count,
                "trial_prediction_index": trial_prediction_index,
                "prediction_index": int(prediction_index),
            },
        }
    )
    return summary


def _summary_for_index(
    summaries: Sequence[object],
    index: int,
) -> Mapping[str, Any] | None:
    for item in summaries:
        if not isinstance(item, Mapping):
            continue
        if int(item.get("index", -1)) == index:
            return item
        if str(item.get("label", "")) == f"R{index:02d}":
            return item
    if index - 1 < len(summaries) and isinstance(summaries[index - 1], Mapping):
        return summaries[index - 1]
    return None


def _format_channel_line(channel: str, values: Mapping[str, Any]) -> str:
    return (
        f"    {channel:<4}: "
        f"P={_fmt(values.get('power'), 6)} | "
        f"B={_fmt(values.get('baseline'), 6)} | "
        f"ERD={_fmt(values.get('erd'), 4, signed=True)}% | "
        f"Drop={_fmt(values.get('power_drop'), 4)}% | "
        f"W={_fmt(values.get('weight'), 1)} | "
        f"C={_fmt(values.get('contribution'), 4, signed=True)}"
    )


def _format_score_items(values: Sequence[object], *, signed: bool = True, digits: int = 6) -> list[str]:
    items = [f"R{index:02d}={_fmt(value, digits, signed=signed)}" for index, value in enumerate(values, start=1)]
    return [" | ".join(items[index: index + 5]) for index in range(0, len(items), 5)]


def _baseline_source_powers(extra: Mapping[str, Any]) -> dict[str, list[float]]:
    powers: dict[str, list[float]] = {channel: [] for channel in BASELINE_REPORT_CHANNELS}
    calibration_power_details = extra.get("calibration_power_details", {})
    if isinstance(calibration_power_details, Mapping):
        for channel in BASELINE_REPORT_CHANNELS:
            details = calibration_power_details.get(channel)
            if isinstance(details, Mapping):
                powers[channel] = [_as_float(value) for value in details.get("trial_powers", [])]
    return powers


def _middle_values(values: Sequence[float]) -> list[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return []
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return [ordered[midpoint]]
    return [ordered[midpoint - 1], ordered[midpoint]]


def _format_power_values(values: Sequence[object]) -> str:
    return " ".join(_fmt(value, 6) for value in values)


def _format_iqr_outliers(outliers: Sequence[object]) -> str:
    if not outliers:
        return "none"
    parts: list[str] = []
    for item in outliers:
        if not isinstance(item, Mapping):
            continue
        parts.append(
            f"{item.get('trial_id', 'unknown')}={_fmt(item.get('power'), 6)} {item.get('reason', 'outlier')}"
        )
    return " | ".join(parts) if parts else "none"


def format_calibration_final_summary_report(profile_payload: Mapping[str, Any] | None) -> str:
    """Format final Rest Power baseline and MI criterion configuration."""
    if not profile_payload:
        return "CALIBRATION FINAL SUMMARY\nnot recorded\n"

    payload = dict(profile_payload)
    extra = dict(payload.get("extra", {}) or {})
    criterion_details = dict(extra.get("criterion_details", {}) or {})
    action_threshold = payload.get("action_threshold", criterion_details.get("T_all"))

    lines = [
        "CALIBRATION FINAL SUMMARY",
        "=" * 60,
        "",
        "PERSONAL REST POWER BASELINE",
        "-" * 60,
        "Calibration only builds per-channel Rest Power baselines.",
        "Prediction later compares live window Power against this personal baseline.",
        "",
        "Rule : one-pass per-channel Tukey 1.5 IQR filter, then median(Clean Rest Powers)",
        "",
    ]

    baselines = dict(payload.get("channel_baselines", {}) or {})
    power_details = dict(extra.get("calibration_power_details", {}) or {})
    for channel in BASELINE_REPORT_CHANNELS:
        final_value = baselines.get(channel)
        details = dict(power_details.get(channel, {}) or {})
        values = list(details.get("trial_powers", []))
        sorted_values = list(details.get("sorted_powers", sorted(_as_float(value) for value in values)))
        clean_values = list(details.get("clean_powers", values))
        clean_sorted = list(details.get("clean_sorted_powers", sorted(_as_float(value) for value in clean_values)))
        clean_middle = _middle_values([_as_float(value) for value in clean_sorted])
        iqr = dict(details.get("iqr_filter", {}) or {})
        if values:
            lines.append(f"{channel}")
            lines.append(f"  Raw Rest Powers    : {_format_power_values(values)}")
            lines.append(f"  Sorted Raw Powers  : {_format_power_values(sorted_values)}")
            lines.append(f"  Q1                 : {_fmt(iqr.get('q1'), 6)}")
            lines.append(f"  Q3                 : {_fmt(iqr.get('q3'), 6)}")
            lines.append(f"  IQR                : Q3 - Q1 = {_fmt(iqr.get('iqr'), 6)}")
            lines.append(f"  Lower Fence        : Q1 - 1.5 * IQR = {_fmt(iqr.get('lower_fence'), 6)}")
            lines.append(f"  Upper Fence        : Q3 + 1.5 * IQR = {_fmt(iqr.get('upper_fence'), 6)}")
            lines.append(f"  IQR Outliers       : {_format_iqr_outliers(list(iqr.get('outliers', [])))}")
            lines.append(f"  Clean Rest Powers  : {_format_power_values(clean_values)}")
            lines.append(f"  Clean Power Count  : {int(details.get('clean_power_count', len(clean_values)))}")
            lines.append(f"  Clean Median Values: {' | '.join(_fmt(value, 6) for value in clean_middle)}")
            lines.append(f"  Baseline      : median(Clean Rest Powers) = {_fmt(final_value, 6)}")
        else:
            lines.append(f"{channel:<4} Baseline = {_fmt(final_value, 6)} | source powers not recorded")
        lines.append("")

    thresholds_without = dict(payload.get("action_thresholds_without_channel", {}) or {})
    lines.extend(
        [
            "",
            "MI DECISION CRITERION CONFIGURATION",
            "-" * 60,
            "Function : get_mi_success_drop_threshold_pct()",
            f"Name     : {criterion_details.get('criterion_name', 'fixed_protocol_mi_success_drop_pct')}",
            f"Version  : {criterion_details.get('criterion_version', 'n/a')}",
            "Meaning  : Selected ROI Power Drop >= threshold means MI.",
            "Source   : fixed protocol criterion, not calculated from Rest ERD.",
            f"T_all    = {_fmt(action_threshold, 6)}%",
        ]
    )
    for channel in ("FC3", "FC4", "C3", "C4", "CP3", "CP4"):
        if channel in thresholds_without:
            lines.append(f"T_no_{channel:<3}= {_fmt(thresholds_without[channel], 6)}%")
    if lines and lines[-1] == "":
        lines.pop()
    lines.append("=" * 60)
    return "\n".join(lines).rstrip() + "\n"


def _format_roi_for_report(
    summary: Mapping[str, Any],
    *,
    title: str,
    score_label: str,
    channels: Sequence[str],
    prefix: str,
) -> list[str]:
    channel_values = dict(summary.get("channels", {}))
    lines = [f"  {title}:"]
    for channel in channels:
        values = channel_values.get(channel)
        if isinstance(values, Mapping):
            lines.append(_format_channel_line(channel, values))
    lines.extend(
        [
            "",
            f"    weighted_sum = {_fmt(summary.get(f'{prefix}_weighted_sum'), 4, signed=True)}",
            f"    weight_sum   = {_fmt(summary.get(f'{prefix}_weight_sum'), 2)}",
            f"    {score_label:<13}= {_fmt(summary.get(f'{prefix}_score'), 4)}",
        ]
    )
    return lines


def format_calibration_loo_report(profile_payload: Mapping[str, Any] | None) -> str:
    if not profile_payload:
        return "未记录\n"
    payload = dict(profile_payload)
    extra = dict(payload.get("extra", {}) or {})
    trial_ids = list(payload.get("calibration_trial_ids", []) or [])
    matrix = extra.get("calibration_power_matrix", {})
    rows = list(matrix.get("rows", []) if isinstance(matrix, Mapping) else [])
    baselines = dict(payload.get("channel_baselines", {}) or {})
    lines = [
        "PERSONAL REST POWER CALIBRATION",
        "=" * 60,
        "说明：这里展示当前正式校准产物：逐 Trial、逐通道 Rest Power，以及按通道取 median 得到的个人静息功率基线。",
        "Prediction 阶段会用实时窗口 Power 与该个人基线计算 ERD / Power Drop。",
        "",
        "Rest Power Matrix",
        "-" * 60,
        f"{'Trial':<8}" + "".join(f"{channel:>12}" for channel in BASELINE_REPORT_CHANNELS),
        "-" * (8 + 12 * len(BASELINE_REPORT_CHANNELS)),
    ]
    if rows:
        for index, row in enumerate(rows, start=1):
            label = f"R{index:02d}"
            trial_id = str(trial_ids[index - 1]) if index - 1 < len(trial_ids) else label
            values = dict(row)
            lines.append(
                f"{label:<8}"
                + "".join(f"{_fmt(values.get(channel), 6):>12}" for channel in BASELINE_REPORT_CHANNELS)
                + f"   ({trial_id})"
            )
    else:
        lines.append("Rest Power Matrix 未记录")
    lines.extend(["", "IQR-Cleaned Median Baseline", "-" * 60])
    for channel in BASELINE_REPORT_CHANNELS:
        source = dict(extra.get("calibration_power_details", {}) or {}).get(channel, {})
        if isinstance(source, Mapping):
            trial_powers = list(source.get("trial_powers", []))
            clean_powers = list(source.get("clean_powers", trial_powers))
            clean_sorted = list(source.get("clean_sorted_powers", sorted(_as_float(value) for value in clean_powers)))
            middle = _middle_values([_as_float(value) for value in clean_sorted])
            outliers = list(dict(source.get("iqr_filter", {}) or {}).get("outliers", []))
            lines.append(
                f"{channel:<4}: IQR clean {len(clean_powers)}/{len(trial_powers)} powers, median = "
                f"{' | '.join(_fmt(value, 6) for value in middle)} -> {_fmt(baselines.get(channel), 6)}"
            )
            if outliers:
                lines.append(f"      outliers: {_format_iqr_outliers(outliers)}")
        else:
            lines.append(f"{channel:<4}: {_fmt(baselines.get(channel), 6)}")
    lines.append("=" * 60)
    return "\n".join(lines).rstrip() + "\n"


def format_prediction_success_window_summary(prediction_result: Mapping[str, Any]) -> str:
    summary = prediction_result.get("report_summary") if isinstance(prediction_result, Mapping) else None
    if not isinstance(summary, Mapping):
        return (
            "成功窗口计算摘要\n"
            "----------------------------------------\n"
            "详细成功窗口计算摘要未记录\n"
        )

    lines = [
        "成功窗口计算摘要",
        "----------------------------------------",
        "P   = Prediction Window Power",
        "B   = Personal Profile Baseline",
        "ERD = (P - B) / B * 100%",
        "Drop = -ERD",
        "W   = Channel Weight",
        "C   = ERD * W",
        "",
        f"Quality          : {summary.get('quality_text', 'n/a')}",
        f"Mode             : {summary.get('quality_mode', 'n/a')}",
        f"Threshold Source : {summary.get('threshold_source', 'n/a')}",
        f"Threshold        : {_fmt(summary.get('action_threshold'), 6)}",
        "",
    ]
    lines.extend(
        _format_roi_for_report(
            summary,
            title="左 ROI",
            score_label="Left Score",
            channels=LEFT_REPORT_CHANNELS,
            prefix="left",
        )
    )
    lines.append("")
    lines.extend(
        _format_roi_for_report(
            summary,
            title="右 ROI",
            score_label="Right Score",
            channels=RIGHT_REPORT_CHANNELS,
            prefix="right",
        )
    )
    lines.extend(
        [
            "",
            "Selected ROI",
            "----------------------------------------",
            (
                f"max(Left Drop={_fmt(summary.get('left_roi_drop_pct', summary.get('left_score')), 6)}%, "
                f"Right Drop={_fmt(summary.get('right_roi_drop_pct', summary.get('right_score')), 6)}%)"
            ),
            f"= {_fmt(summary.get('selected_roi_drop_pct', summary.get('action_score')), 6)}%",
            f"Selected ROI = {summary.get('selected_roi', 'n/a')}",
            f"action_score = selected_roi_drop_pct = {_fmt(summary.get('action_score'), 6)}",
            "",
            "Threshold",
            "----------------------------------------",
            f"{summary.get('threshold_source', 'n/a')} = {_fmt(summary.get('action_threshold'), 6)}%",
            "",
            "Final Decision",
            "----------------------------------------",
            (
                f"{_fmt(summary.get('selected_roi_drop_pct', summary.get('action_score')), 6)}% "
                f">= {_fmt(summary.get('action_threshold'), 6)}%"
            ),
            f"criterion_met = {bool(dict(summary.get('decision_detail', {}) or {}).get('criterion_met', False))}",
            f"margin = {_fmt(summary.get('score_margin'), 6, signed=True)}",
            "",
            f"算法窗口结果 : {summary.get('algorithm_window_result', 'n/a')}",
            f"最终 display_code : {prediction_result.get('display_code', 'n/a')}",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def format_window_identity_for_report(prediction_result: Mapping[str, Any]) -> dict[str, object]:
    summary = prediction_result.get("report_summary") if isinstance(prediction_result, Mapping) else None
    identity = summary.get("window_identity", {}) if isinstance(summary, Mapping) else {}
    if not isinstance(identity, Mapping):
        identity = {}
    backend_trial_count = identity.get("backend_trial_count")
    trial_prediction_index = identity.get("trial_prediction_index")
    prediction_index = identity.get("prediction_index", prediction_result.get("prediction_index", "N/A"))
    return {
        "backend_trial_count": _fmt_optional(backend_trial_count),
        "trial_prediction_index": _fmt_optional(trial_prediction_index),
        "prediction_index": _fmt_optional(prediction_index),
    }
