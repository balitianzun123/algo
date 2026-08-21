"""Single-trial inference for the fixed-rule hierarchical algorithm."""

# 使用 profile 完成最终判断预测
# 使用Profile进行单trial推理

from __future__ import annotations

import re
import numpy as np

from .constants import ACTION_LABEL, PROFILE_SCHEMA_VERSION, REST_LABEL
from .exceptions import InvalidProfileError
from .features import extract_trial_features
from .models import PersonalProfile, PredictionResult, TrialInput

# mode 必须是 cal10 或 rest10
# 采样率和阈值必须是有限数值
# baseline 不能为空
# 每个 baseline 必须大于 0
def _validate_profile(profile: PersonalProfile) -> None:
    # 兼容既有 cal10/rest10，同时允许新的 rest<N> profile。
    if profile.mode != "cal10" and re.fullmatch(r"rest[1-9][0-9]*", profile.mode) is None:
        raise InvalidProfileError(f"Unsupported profile mode: {profile.mode}")
    if str(getattr(profile, "profile_schema_version", "")) != PROFILE_SCHEMA_VERSION:
        raise InvalidProfileError(
            "Profile schema version is incompatible with the current Power Drop algorithm; "
            "profile requires recalibration."
        )
    values = [profile.sampling_rate, profile.action_threshold, profile.laterality_threshold, profile.window_start_s, profile.window_end_s]
    if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        raise InvalidProfileError("Profile numeric fields must be finite.")
    if not profile.channel_baselines:
        raise InvalidProfileError("Profile channel_baselines must not be empty.")
    for channel, value in profile.channel_baselines.items():
        if not np.isfinite(float(value)) or float(value) <= 0:
            raise InvalidProfileError(f"Profile baseline for {channel} must be positive and finite.")
    for channel, value in dict(profile.action_thresholds_without_channel or {}).items():
        if not np.isfinite(float(value)):
            raise InvalidProfileError(
                f"Profile single-channel-exclusion threshold for {channel} must be finite."
            )


def predict_trial(
    *,
    trial: TrialInput,
    profile: PersonalProfile,
) -> PredictionResult:
    """Predict rest, left, or right for one EEG trial using a frozen profile."""
    _validate_profile(profile)
    if tuple(trial.channel_names) != tuple(profile.channel_names):
        raise InvalidProfileError("Trial channel_names must match profile channel_names.")
    if float(trial.sampling_rate) != float(profile.sampling_rate):
        raise InvalidProfileError("Trial sampling_rate must match profile sampling_rate.")
    # 调用
    # channel_powers
    # channel_erd
    # action_score
    # laterality_score
    features = extract_trial_features(trial=trial, channel_baselines=profile.channel_baselines)
    if float(features.action_score) < float(profile.action_threshold):
        return PredictionResult(
            final_result=REST_LABEL,
            stage1_result=REST_LABEL,
            stage2_result=None,
            action_score=float(features.action_score),
            action_threshold=float(profile.action_threshold),
            laterality_score=float(features.laterality_score),
            laterality_threshold=float(profile.laterality_threshold),
            channel_powers={
                str(channel): float(value)
                for channel, value in features.channel_powers.items()
            },
            channel_erd={
                str(channel): float(value)
                for channel, value in features.channel_erd.items()
            },
            action_score_details=dict(features.action_score_details),
        )
    return PredictionResult(
        final_result=ACTION_LABEL,
        stage1_result=ACTION_LABEL,
        stage2_result=None,
        action_score=float(features.action_score),
        action_threshold=float(profile.action_threshold),
        laterality_score=float(features.laterality_score),
        laterality_threshold=float(profile.laterality_threshold),
        channel_powers={
            str(channel): float(value)
            for channel, value in features.channel_powers.items()
        },
        channel_erd={
            str(channel): float(value)
            for channel, value in features.channel_erd.items()
        },
        action_score_details=dict(features.action_score_details),
    )


def predict_from_scores(
    *,
    action_score: float,
    laterality_score: float,
    action_threshold: float,
    laterality_threshold: float,
) -> PredictionResult:
    """Predict from already-computed scores without changing thresholds."""
    if float(action_score) < float(action_threshold):
        return PredictionResult(
            final_result=REST_LABEL,
            stage1_result=REST_LABEL,
            stage2_result=None,
            action_score=float(action_score),
            action_threshold=float(action_threshold),
            laterality_score=float(laterality_score),
            laterality_threshold=float(laterality_threshold),
        )
    return PredictionResult(
        final_result=ACTION_LABEL,
        stage1_result=ACTION_LABEL,
        stage2_result=None,
        action_score=float(action_score),
        action_threshold=float(action_threshold),
        laterality_score=float(laterality_score),
        laterality_threshold=float(laterality_threshold),
    )
