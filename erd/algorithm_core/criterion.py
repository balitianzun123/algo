"""Transparent MI decision criterion configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

MI_SUCCESS_DROP_THRESHOLD_PCT: float = 20.0
MI_SUCCESS_CRITERION_NAME: str = "fixed_protocol_mi_success_drop_pct"
MI_SUCCESS_CRITERION_VERSION: str = "1.0.0"


def get_mi_success_drop_threshold_pct(
    *,
    subject_id: str | None = None,
    excluded_channel: str | None = None,
    used_channels: Sequence[str] | None = None,
    channel_baselines: Mapping[str, float] | None = None,
    calibration_power_details: Mapping[str, object] | None = None,
) -> float:
    """Return the required ROI Power Drop percentage for MI success.

    The current protocol criterion is a simple fixed 20% ROI Power Drop.
    It is not learned from Rest calibration statistics or any adaptive
    statistic. Arguments are reserved so the same single entry point
    can later handle subject-specific or excluded-channel protocol rules.
    """

    _ = subject_id, excluded_channel, used_channels, channel_baselines, calibration_power_details
    return float(MI_SUCCESS_DROP_THRESHOLD_PCT)
