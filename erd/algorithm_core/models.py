"""Dataclasses used by the product-callable algorithm core."""

# 定义 Trial、Profile、Features、PredictionResult 等数据结构
#  Trial、Profile、Features、PredictionResult数据结构
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .constants import PROFILE_SCHEMA_VERSION, WINDOW_END_S, WINDOW_START_S, ANALYSIS_BAND_HZ


@dataclass(frozen=True)
class RawTrialInput:
    """Raw or event-aligned EEG input before the legacy 50 Hz + 8-35 Hz preprocessing.

    The data array is interpreted as [channels, samples].  The unit is not
    converted by the legacy dataset pipeline, so callers must explicitly mark
    the numeric values as preserved raw NPY units.
    """

    trial_id: str
    data: np.ndarray
    channel_names: tuple[str, ...]
    sampling_rate: float
    data_unit: str
    label: str | None = None
    event_sample: int | None = None
    subject_id: str | None = None
    run_id: str | None = None
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class TrialInput:
    """Single EEG trial, shaped [channels, samples]."""

    trial_id: str
    data: np.ndarray
    channel_names: tuple[str, ...]
    sampling_rate: float
    label: str | None = None
    preprocess_summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialFeatures:
    """Fixed ERD features for one trial."""

    channel_powers: dict[str, float]
    channel_erd: dict[str, float]
    action_score: float
    laterality_score: float
    action_score_details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PersonalProfile:
    """Personal calibration profile for Cal-10 or Rest-10 inference."""

    profile_id: str
    subject_id: str
    mode: str
    sampling_rate: float
    channel_names: tuple[str, ...]
    channel_baselines: dict[str, float]
    action_threshold: float
    laterality_threshold: float
    threshold_source_action: str
    threshold_source_laterality: str
    action_thresholds_without_channel: dict[str, float] = field(default_factory=dict)
    created_at: str | None = None
    window_start_s: float = WINDOW_START_S
    window_end_s: float = WINDOW_END_S
    analysis_band_hz: tuple[float, float] = ANALYSIS_BAND_HZ
    calibration_trial_ids: tuple[str, ...] = field(default_factory=tuple)
    profile_schema_version: str = PROFILE_SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionResult:
    """Three-class prediction result for one trial."""

    final_result: str
    stage1_result: str
    stage2_result: str | None
    action_score: float
    action_threshold: float
    laterality_score: float | None
    laterality_threshold: float
    
    # 仅用于诊断/分析，不改变预测判定逻辑。
    channel_powers: dict[str, float] = field(default_factory=dict)
    channel_erd: dict[str, float] = field(default_factory=dict)
    action_score_details: dict[str, Any] = field(default_factory=dict)
