"""Unified backend entry point for raw-EEG Rest-N calibration and prediction.

`filtered_samples` is a legacy compatibility parameter name. The backend passes
raw EEG shaped `[time_steps, backend_channels]`; Python performs channel
selection, 50 Hz notch, 8-30 Hz band-pass, and a 0-4 s trial crop internally.
"""

# 整个流程

# 新用户
#   ↓
# 静息校准
#   ↓
# 生成个人 profile
#   ↓
# 使用个人 profile 预测 动/不动

# 可以指定做几次：required_rest_trials=5

# 同时可以支持免个人校准预测 即当前用户不用现场进行个人校准，但使用另一份已经验证过的通用 profile。

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from algo.erd.predict import predict_rest10_stream
from algo.erd.train import DEFAULT_PROFILE_DIR, RAW_ADC_OFFSET, RAW_ADC_SCALE, reset_session, train_rest10_stream

# 设备编号从 0 开始；算法层从完整 16 通道数据选取并重排。
DEVICE_STANDARD_MAPPING = {
    "FC3": 2, "FC4": 5, "C3": 8, "C4": 10,
    "CP3": 13, "CPz": 12, "CP4": 11, "Cz": 9,
}
STANDARD_MAPPING = DEVICE_STANDARD_MAPPING
# 当前运行时配置
ALGORITHM_RUNTIME_CONFIG = {
    "required_rest_trials": 10,  # 静息校准需要收集的有效 trial 数。
    "prediction_step_seconds": 1,  # 预测滑动步长，单位秒；1.0 表示每秒输出一次预测。
    "live_quality_window_seconds": 4.0,  # 实时通道质量评估使用的滚动窗口长度，单位秒。
    "live_quality_step_seconds": 1.0,  # 实时通道质量评估刷新步长，单位秒。
    "live_quality_filter_warmup_seconds": 0.5,  # 实时质量连续滤波的预热时长，单位秒。
}


# train 和 predict 的统一分发入口
def sim_demo(
        case_no: str,
        age: int,
        gender: int,
        sample_rate: int,
        # 静息为 type = 0，预测则为 1
        type: int,
        # bus_type=0：静息
        bus_type: int,
        filtered_samples: np.ndarray,# 暂时忽略
        STANDARD_MAPPING: Mapping[str, int],
        *,
        profile_dir: str | Path = DEFAULT_PROFILE_DIR,
        # 设置需要校准的trial次数，默认10次
        required_rest_trials: int = 10,
        #默认不允许不校准
        # 只有以下条件同时成立时才使用：
        # 个人 profile 不存在
        # allow_uncalibrated_prediction=True
        # fallback_profile_path 已提供
        # fallback 文件存在且通过校验
        allow_uncalibrated_prediction: bool = False,
        fallback_profile_path: str | Path | None = None,
        prediction_step_seconds: float = 1.0,
        backend_trial_count: int | None = None,
) -> dict[str, Any]:
    """Dispatch the sole backend entry point without changing positional calls.

    `type=0` buffers raw rest data until it can produce a personal Rest-N
    profile. `type=1` predicts rest/action. By default prediction requires that
    personal profile; an uncalibrated prediction additionally requires an
    explicit caller-owned fallback profile path.
    """
    try:
        # 统一入口只负责按 type 分流，具体缓存和算法逻辑留在适配层。
        if int(type) == 0:#静息
            return train_rest10_stream(
                case_no=case_no, age=age, gender=gender, sample_rate=sample_rate, type=type,
                bus_type=bus_type, filtered_samples=filtered_samples, standard_mapping=STANDARD_MAPPING,
                profile_dir=profile_dir, required_rest_trials=required_rest_trials,
            )
        if int(type) == 1:#预测
            return predict_rest10_stream(
                case_no=case_no, sample_rate=sample_rate, type=type, filtered_samples=filtered_samples,
                standard_mapping=STANDARD_MAPPING, profile_dir=profile_dir,
                required_rest_trials=required_rest_trials,
                allow_uncalibrated_prediction=allow_uncalibrated_prediction,
                fallback_profile_path=fallback_profile_path,
                prediction_step_seconds=prediction_step_seconds,
                backend_trial_count=backend_trial_count,
            )
        return {
            "success": False, "code": -1, "status": "unsupported_type", "case_no": str(case_no),
            "message": f"unsupported type={type}; expected 0 calibration or 1 prediction.",
        }
    except Exception as exc:
        return {"success": False, "code": -1, "status": "error", "case_no": str(case_no), "message": str(exc)}


def _synthetic_backend_batch(
        *, amplitude: float, sample_rate: int = 2000, samples: int | None = None, n_channels: int = 16,
) -> np.ndarray:
    """Create unsigned-ADC `[time, channels]` demo EEG with mapped 10 Hz content."""
    trial_samples = int(round(4.0 * float(sample_rate))) if samples is None else int(samples)
    time = np.arange(trial_samples, dtype=np.float64) / float(sample_rate)
    signal = amplitude * np.sin(2.0 * np.pi * 10.0 * time)
    batch = np.zeros((trial_samples, n_channels), dtype=np.float64)
    for index in STANDARD_MAPPING.values():
        batch[:, index] = signal
    # 演示数据遵循后端 ADC 原始码协议，进入 sim_demo 后再统一换算。
    return RAW_ADC_OFFSET + batch / RAW_ADC_SCALE

# 免校准的逻辑
# 先查当前用户个人 profile
#     ↓
# 存在：使用个人 profile
#     ↓
# 不存在：检查 allow_uncalibrated_prediction
#     ↓
# 检查 fallback_profile_path
#     ↓
# 加载并验证 fallback
#     ↓
# 收满 4 秒原始 EEG
#     ↓
# 统一预处理
#     ↓
# 预测

if __name__ == "__main__":
    # demo 与正式后端使用同一个默认 profile 目录。
    demo_case_no, demo_profile_dir = "demo_case_rest10", DEFAULT_PROFILE_DIR
    reset_session(demo_case_no)
    for trial_index in range(10):
        result = sim_demo(
            demo_case_no, 60, 1, 2000, 0, 0, _synthetic_backend_batch(amplitude=10.0), STANDARD_MAPPING,
            profile_dir=demo_profile_dir,
        )
        print(f"default 10-trial calibration {trial_index + 1}: {result}")
    result = sim_demo(
        demo_case_no, 60, 1, 2000, 1, 0, _synthetic_backend_batch(amplitude=0.1), STANDARD_MAPPING,
        profile_dir=demo_profile_dir,
    )
    print(f"personal-profile prediction: {result}")
    print("custom calibration: sim_demo(..., type=0, required_rest_trials=5)")
    no_profile = sim_demo(
        "demo_no_profile", 60, 1, 2000, 1, 0, _synthetic_backend_batch(amplitude=0.1), STANDARD_MAPPING,
        profile_dir=demo_profile_dir,
    )
    print(f"default uncalibrated prediction is blocked: {no_profile}")
    fallback_result = sim_demo(
        "demo_fallback", 60, 1, 2000, 1, 0, _synthetic_backend_batch(amplitude=0.1), STANDARD_MAPPING,
        profile_dir=demo_profile_dir, allow_uncalibrated_prediction=True,
        fallback_profile_path=demo_profile_dir / "caller_validated_profile.json",
    )
    print(f"explicit fallback path is required to exist and validate: {fallback_result}")
