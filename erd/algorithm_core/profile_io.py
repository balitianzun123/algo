"""JSON profile persistence for the algorithm core."""

# 保存和读取 profile JSON

from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any

from .exceptions import InvalidProfileError, ProfileAlreadyExistsError
from .inference import _validate_profile
from .models import PersonalProfile

# 完成校准后，会生成类似：

# patient_001_rest5_profile.json
# 会生成以下内容：
# 用户编号
# profile ID
# 模式 rest5
# profile 保存实际采样率；当前设备正式流程使用 2000 Hz。
# 通道顺序
# 各通道静息 baseline
# 动作阈值
# 左右阈值
# 校准 trial ID
# required_rest_trials=5
# 5 个留一法静息动作分数
# 算法版本等信息
def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def save_profile(profile: PersonalProfile, output_path: str | Path, *, overwrite: bool = False) -> None:
    """Save a profile as UTF-8 JSON using atomic replacement."""
    _validate_profile(profile)
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise ProfileAlreadyExistsError(f"Profile already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dataclasses.asdict(profile)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    temp.replace(path)


def load_profile(profile_path: str | Path) -> PersonalProfile:
    """Load and validate a JSON profile."""
    path = Path(profile_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise InvalidProfileError(f"Cannot load profile JSON: {path}") from exc
    required = {
        "profile_id",
        "subject_id",
        "mode",
        "sampling_rate",
        "channel_names",
        "channel_baselines",
        "action_threshold",
        "laterality_threshold",
        "threshold_source_action",
        "threshold_source_laterality",
    }
    missing = required - set(payload)
    if missing:
        raise InvalidProfileError(f"Profile is missing required fields: {sorted(missing)}")
    try:
        profile = PersonalProfile(
            profile_id=str(payload["profile_id"]),
            subject_id=str(payload["subject_id"]),
            mode=str(payload["mode"]),
            sampling_rate=float(payload["sampling_rate"]),
            channel_names=tuple(str(x) for x in payload["channel_names"]),
            channel_baselines={str(k): float(v) for k, v in dict(payload["channel_baselines"]).items()},
            action_threshold=float(payload["action_threshold"]),
            laterality_threshold=float(payload["laterality_threshold"]),
            threshold_source_action=str(payload["threshold_source_action"]),
            threshold_source_laterality=str(payload["threshold_source_laterality"]),
            action_thresholds_without_channel={
                str(k): float(v)
                for k, v in dict(payload.get("action_thresholds_without_channel", {})).items()
            },
            created_at=payload.get("created_at"),
            window_start_s=float(payload.get("window_start_s", 0.5)),
            window_end_s=float(payload.get("window_end_s", 2.5)),
            analysis_band_hz=tuple(float(x) for x in payload.get("analysis_band_hz", (8.0, 30.0))),
            calibration_trial_ids=tuple(str(x) for x in payload.get("calibration_trial_ids", ())),
            profile_schema_version=str(payload.get("profile_schema_version", "1.0.0")),
            extra=dict(payload.get("extra", {})),
        )
    except Exception as exc:
        raise InvalidProfileError(f"Profile has invalid field values: {path}") from exc
    if any(not math.isfinite(float(x)) for x in (profile.sampling_rate, profile.action_threshold, profile.laterality_threshold)):
        raise InvalidProfileError("Profile numeric fields must be finite.")
    _validate_profile(profile)
    return profile
