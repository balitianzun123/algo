"""脑电 TCP 数据与算法主入口之间的桥接层。"""

from __future__ import annotations

import logging
import time
import importlib
import math
import threading
from collections.abc import Mapping
from functools import wraps
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import signal

from algo.erd.algorithm_core.constants import BANDPOWER_METHOD, FILTER_ORDER, PREDICTION_INVALID_QUALITY
from algo.erd.algorithm_core import trace as eeg_trace
from algo.erd.algorithm_core.preprocessing import apply_notch_filter, apply_bandpass_filter
from algo.erd.enter import ALGORITHM_RUNTIME_CONFIG, DEVICE_STANDARD_MAPPING, sim_demo
from algo.erd.regulatory.bandpower import compute_bandpower
from algo.erd.train import DEFAULT_PROFILE_DIR, RAW_ADC_OFFSET, RAW_ADC_SCALE, reset_session, profile_path_for_case, get_session

DEVICE_CHANNEL_COUNT = 16
ADC_COUNT_MAX = 16777215.0
ADC_RAIL_VALUES = (0.0, ADC_COUNT_MAX)
PROTOCOL_CHANNEL_NAMES = {index: name for name, index in DEVICE_STANDARD_MAPPING.items()}
ACTIVE_PROTOCOL_CHANNELS = tuple(sorted(PROTOCOL_CHANNEL_NAMES))
LIVE_CHANNEL_NAMES = tuple(DEVICE_STANDARD_MAPPING)
REST_LABEL = "rest"
ACTION_LABEL = "action"
TOPO_MAP_COORDS = {
    "FC3": (-0.45, 0.55),
    "FC4": (0.45, 0.55),
    "C3": (-0.55, 0.10),
    "C4": (0.55, 0.10),
    "CP3": (-0.45, -0.45),
    "CPz": (0.00, -0.50),
    "CP4": (0.45, -0.45),
    "Cz": (0.00, 0.05),
}
TOPO_MAP_BANDS = {
    "alpha": (8.0, 13.0),  # α 脑地图频段，单位 Hz；只用于脑地图，不影响训练/预测特征。
    "beta": (13.0, 30.0),  # β 脑地图频段，单位 Hz；只用于脑地图，不影响训练/预测特征。
}
TOPO_MAP_BAND_TITLES = {
    "alpha": "α 波 8-13 Hz",
    "beta": "β 波 13-30 Hz",
}
_impedance_module: Any | None = None
_impedance_import_error: str | None = None

logger = logging.getLogger(__name__)

def _configured_value(config: Mapping[str, Any], key: str, fallback: Any) -> Any:
    """从运行配置中读取参数；没有配置时使用传入的默认值。"""
    return config[key] if key in config else fallback


def _load_impedance_module() -> Any | None:
    """按需导入 impedance 扩展；只在实时通道质量评估需要时加载。"""
    global _impedance_module, _impedance_import_error
    if _impedance_module is not None:
        return _impedance_module
    try:
        _impedance_module = importlib.import_module("impedance")
        _impedance_import_error = None
    except Exception as exc:
        _impedance_import_error = str(exc)
        return None
    return _impedance_module


def _serialized_feed(method):
    """给 feed 加锁，避免多个 TCP 线程同时写入算法缓存。"""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        """在持有实例锁的情况下调用原始 feed 方法。"""
        with self._feed_lock:
            return method(self, *args, **kwargs)
    return wrapped


class EEGAlgorithmBridge:
    """实时脑电算法桥接器。

    该类接收后端解析出的 16 通道 unsigned ADC count，先转换成微伏，
    再更新实时通道质量与脑地图，并最终调用 sim_demo。
    为兼容原有 train/predict 入口，微伏数据会临时换回 ADC-like 格式。
    """

    def __init__(
            self,
            *,
            case_no: str = "patient_001",
            age: int = 30,
            gender: int = 1,
            sample_rate: int = 2000,
            required_rest_trials: int = 10,
            prediction_step_seconds: float = 1.0,
            # 前 0.5秒 不用于 baseline
            baseline_skip_seconds: float = 0.0,
            baseline_calibration_seconds: float = 3.0,
            live_quality_window_seconds: float = 3.0,
            live_quality_step_seconds: float = 1.0,
            live_quality_filter_warmup_seconds: float = 0.5,
            prediction_smoothing: Mapping[str, Any] | None = None,
            profile_dir: str | Path = DEFAULT_PROFILE_DIR,
            runtime_config: Mapping[str, Any] | None = ALGORITHM_RUNTIME_CONFIG,
    ) -> None:
        print("[DEBUG] NEW EEGAlgorithmBridge:",id(self))
        """初始化桥接器，并用 main.py 的统一配置覆盖后端传入的旧参数。"""
        config = {} if runtime_config is None else runtime_config
        required_rest_trials = _configured_value(config, "required_rest_trials", required_rest_trials)
        prediction_step_seconds = _configured_value(config, "prediction_step_seconds", prediction_step_seconds)
        _ = baseline_skip_seconds, baseline_calibration_seconds, prediction_smoothing
        live_quality_window_seconds = _configured_value(
            config,
            "live_quality_window_seconds",
            live_quality_window_seconds,
        )
        live_quality_step_seconds = _configured_value(
            config,
            "live_quality_step_seconds",
            live_quality_step_seconds,
        )
        live_quality_filter_warmup_seconds = _configured_value(
            config,
            "live_quality_filter_warmup_seconds",
            live_quality_filter_warmup_seconds,
        )

        self.case_no = str(case_no)
        self.age = int(age)
        self.gender = int(gender)
        self.sample_rate = int(sample_rate)
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        self.required_rest_trials = int(required_rest_trials)
        self.prediction_step_seconds = float(prediction_step_seconds)
        self.live_quality_window_seconds = float(live_quality_window_seconds)
        self.live_quality_step_seconds = float(live_quality_step_seconds)
        if not np.isfinite(self.live_quality_window_seconds) or self.live_quality_window_seconds < 0:
            raise ValueError("live_quality_window_seconds must be finite and nonnegative.")
        if not np.isfinite(self.live_quality_step_seconds) or self.live_quality_step_seconds <= 0:
            raise ValueError("live_quality_step_seconds must be finite and positive.")
        self.live_quality_sample_count = int(round(self.live_quality_window_seconds * self.sample_rate))
        self.live_quality_step_samples = max(1, int(round(self.live_quality_step_seconds * self.sample_rate)))
        self.live_quality_filter_warmup_seconds = float(live_quality_filter_warmup_seconds)
        if (
                not np.isfinite(self.live_quality_filter_warmup_seconds)
                or self.live_quality_filter_warmup_seconds < 0
        ):
            raise ValueError("live_quality_filter_warmup_seconds must be finite and nonnegative.")
        self.live_quality_filter_warmup_sample_count = int(
            round(self.live_quality_filter_warmup_seconds * self.sample_rate)
        )
        self._live_quality_buffer: list[np.ndarray] = []
        self._live_quality_since_update = 0
        self._initialize_live_quality_filters()
        if self.live_quality_filter_warmup_sample_count <= 0:
            self._live_quality_filter_ready = True
        self._live_quality_status = self._make_live_quality_status("collecting", False)
        self._prediction_result_enabled = False
        self._prediction_success_latched = False
        self.profile_dir = Path(profile_dir)
        self.block_samples = self.sample_rate
        self.mode = 0
        self._profile_ready = False
        self._latest_profile_path: Path | None = None
        self._sample_buffer: list[np.ndarray] = []
        self.topomap_window_seconds = self.live_quality_window_seconds
        self.topomap_sample_count = int(round(self.topomap_window_seconds * self.sample_rate))
        self._session_topomap_buffer: list[np.ndarray] = []
        self._trace_adc_seen_samples = 0
        self._trace_adc_stride_samples = max(1, int(round(float(self.sample_rate))))
        self._backend_trial_count_for_trace: int | None = None
        self._session_topomap_bandpowers: dict[str, list[np.ndarray]] = {
            band_name: [] for band_name in TOPO_MAP_BANDS
        }  # 每个频段单独累计整场窗口功率；每个元素形状为 [8]，单位为功率值。
        self._feed_lock = threading.Lock()

    def _initialize_live_quality_filters(self) -> None:
        b_notch, a_notch = signal.iirnotch(
            w0=50.0,
            Q=30.0,
            fs=float(self.sample_rate),
        )
        self._live_quality_notch_sos = signal.tf2sos(
            b_notch,
            a_notch,
        )
        self._live_quality_bandpass_sos = signal.butter(
            4,
            [8.0, 30.0],
            btype="bandpass",
            fs=float(self.sample_rate),
            output="sos",
        )
        live_channel_count = len(LIVE_CHANNEL_NAMES)
        self._live_quality_notch_zi = np.zeros(
            (self._live_quality_notch_sos.shape[0], live_channel_count, 2),
            dtype=np.float64,
        )
        self._live_quality_bandpass_zi = np.zeros(
            (self._live_quality_bandpass_sos.shape[0], live_channel_count, 2),
            dtype=np.float64,
        )
        self._live_quality_filter_warmup_seen = 0
        self._live_quality_filter_ready = False

    def _reset_prediction_state(self) -> None:
        """Deprecated no-op; formal prediction no longer smooths or votes."""
        return None

    def _reset_prediction_result_latch(self, *, enabled: bool = False) -> None:
        """Compatibility shell; display_code now always mirrors the current window code."""
        self._prediction_result_enabled = bool(enabled)
        self._prediction_success_latched = False

    def set_backend_trial_count(self, count: object) -> None:
        """Store backend trial count for trace headers only."""
        try:
            normalized = None if count is None else int(count)
        except (TypeError, ValueError, OverflowError):
            normalized = None
        with self._feed_lock:
            self._backend_trial_count_for_trace = normalized

    def _expected_profile_path(self) -> Path:
        """根据当前受试者编号和静息 trial 数，计算预测阶段应读取的个人 profile 路径。"""
        return profile_path_for_case(
            self.case_no,
            self.profile_dir,
            required_rest_trials=self.required_rest_trials,
        )
    # 校准接口
    def start_calibration(self) -> dict[str, Any]:
        """后端手动进入静息校准/训练阶段；会清空训练缓存，但不删除已经保存的 profile 文件。"""
        with self._feed_lock:
            reset_session(self.case_no)
            self.mode = 0
            self._profile_ready = False
            self._latest_profile_path = None
            self._sample_buffer.clear()
            self._backend_trial_count_for_trace = None
            self._reset_prediction_state()
            self._reset_prediction_result_latch(enabled=False)
            return {
                "success": True,
                "status": "calibration_started",
                "mode": int(self.mode),
                "message": "Calibration mode started; feed() will build a profile and then wait for start_prediction().",
            }
    # 预测接口
    def start_prediction(self) -> dict[str, Any]:
        """后端手动进入预测阶段；只有当前 case 的 profile 已生成或已存在时才允许切换。"""

        with self._feed_lock:
            profile_path = self._latest_profile_path if self._latest_profile_path is not None else self._expected_profile_path()
            if not self._profile_ready and not profile_path.exists():
                return {
                    "success": False,
                    "status": "profile_not_ready",
                    "mode": int(self.mode),
                    "profile_path": str(profile_path),
                    "message": "Profile is not ready; run calibration before starting prediction.",
                }
            if not profile_path.exists():
                return {
                    "success": False,
                    "status": "profile_file_missing",
                    "mode": int(self.mode),
                    "profile_path": str(profile_path),
                    "message": "Profile was marked ready, but the profile file does not exist.",
                }

            self.mode = 1
            self._profile_ready = True
            self._latest_profile_path = profile_path
            self._sample_buffer.clear()
            self._backend_trial_count_for_trace = None
            prediction_session = get_session(
                self.case_no,
                age=self.age,
                gender=self.gender,
                required_rest_trials=self.required_rest_trials,
            )
            prediction_session.prediction_backend_trial_count = None
            prediction_session.trial_prediction_index = 0
            self._session_topomap_buffer.clear()
            for bandpowers in self._session_topomap_bandpowers.values():
                bandpowers.clear()
            self._reset_prediction_state()
            self._reset_prediction_result_latch(enabled=False)
            return {
                "success": True,
                "status": "prediction_started",
                "mode": int(self.mode),
                "profile_path": str(profile_path),
                "message": "Prediction mode started; subsequent feed() blocks will call predict.",
            }
    #锁存状态
    def start_using_prediction_result(self) -> dict[str, Any]:
        """Compatibility API; current display_code always mirrors current window code."""
        with self._feed_lock:
            if self.mode != 1:
                return {
                    "success": False,
                    "status": "prediction_not_started",
                    "mode": int(self.mode),
                    "prediction_result_enabled": bool(self._prediction_result_enabled),
                    "success_latched": bool(self._prediction_success_latched),
                    "message": "Call start_prediction() before using prediction results.",
                }
            self._reset_prediction_result_latch(enabled=True)
            return {
                "success": True,
                "status": "prediction_result_started",
                "mode": int(self.mode),
                "prediction_result_enabled": True,
                "success_latched": False,
                "message": "Prediction result display is enabled; display_code mirrors the current window code.",
            }

    def reset_prediction_result_latch(self) -> dict[str, Any]:
        """Compatibility API; there is no success latch in the current algorithm."""
        with self._feed_lock:
            self._reset_prediction_result_latch(enabled=False)
            return {
                "success": True,
                "status": "prediction_result_latch_reset",
                "mode": int(self.mode),
                "prediction_result_enabled": False,
                "success_latched": False,
                "message": "Prediction result display latch has been reset.",
            }

    def get_signal_calibration_status(self) -> dict[str, Any]:
        """Deprecated DC-baseline status endpoint retained for backend compatibility."""
        with self._feed_lock:
            channels = []
            for channel_index in ACTIVE_PROTOCOL_CHANNELS:
                channels.append({
                    "channel": int(channel_index),
                    "name": PROTOCOL_CHANNEL_NAMES[channel_index],
                    "active": True,
                    "status": "not_used",
                    "invalid_samples": 0,
                    "baseline": None,
                })

            return {
                "status": "disabled",
                "ready": True,
                "message": "Time-domain DC offset baseline is not used by the current algorithm.",
                "baseline_samples": 0,
                "required_baseline_samples": 0,
                "baseline_calibration_seconds": 0.0,
                "active_protocol_channels": [int(index) for index in ACTIVE_PROTOCOL_CHANNELS],
                "channels": channels,
            }
    # 通道实时显示
    def reset_session_topomap(self) -> dict[str, Any]:
        """清空整场平均脑地图累计数据；新一场在线实验开始前可由后端调用。"""
        with self._feed_lock:
            self._session_topomap_buffer.clear()
            for bandpowers in self._session_topomap_bandpowers.values():
                bandpowers.clear()
            return self._make_session_topomap_status("reset", False)

    def get_session_bandpower_topomap_image(self) -> dict[str, Any]:
        """生成整场平均 α/β bandpower 双脑地图 PNG，并返回图片路径给后端。"""
        # 防止一边 feed 正在写脑地图数据，另一边后端又同时来生成图片，导致数据读写冲突。
        with self._feed_lock:
            if not all(self._session_topomap_bandpowers[band_name] for band_name in TOPO_MAP_BANDS):
                return self._make_session_topomap_status("collecting", False)
            mean_power_by_band = {
                band_name: np.mean(np.asarray(self._session_topomap_bandpowers[band_name], dtype=np.float64), axis=0)
                for band_name in TOPO_MAP_BANDS
            }  # dict[band_name] -> [8]，每个频段把整场所有窗口先平均成 8 个通道功率。

        mean_power_db_by_band = {
            band_name: 10.0 * np.log10(np.maximum(mean_power, np.finfo(np.float64).tiny))
            for band_name, mean_power in mean_power_by_band.items()
        }  # dict[band_name] -> [8]，功率转 dB 后用于绘图和后端展示。

        #返回结构化数据
        channels_by_band = {
            band_name: self._topomap_channel_payload(mean_power_db)
            for band_name, mean_power_db in mean_power_db_by_band.items()
        }
        image_path = self.profile_dir / f"{self._safe_case_no()}_session_topomap_alpha_beta.png"
        try:
            self._render_bandpower_topomap(mean_power_db_by_band, image_path)
        except Exception as exc:
            return self._make_session_topomap_status("error", False, channels_by_band=channels_by_band, message=str(exc))

        return self._make_session_topomap_status(
            "ready",
            True,
            channels_by_band=channels_by_band,
            image_path=str(image_path),
        )

    def _safe_case_no(self) -> str:
        """把受试者编号转换成适合文件名使用的字符串。"""
        safe = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in self.case_no)
        return safe or "patient"

    def _make_session_topomap_status(
            self,
            status: str,
            ready: bool,
            *,
            channels_by_band: dict[str, list[dict[str, Any]]] | None = None,
            image_path: str | None = None,
            message: str | None = None,
    ) -> dict[str, Any]:
        """生成统一格式的整场平均脑地图状态字典。"""
        segments_by_band = {
            band_name: int(len(self._session_topomap_bandpowers[band_name]))
            for band_name in TOPO_MAP_BANDS
        }
        result: dict[str, Any] = {
            "status": status,
            "ready": bool(ready),
            "image_path": image_path,
            "bands": {
                band_name: [float(band_range[0]), float(band_range[1])]
                for band_name, band_range in TOPO_MAP_BANDS.items()
            },
            "value_type": "mean_bandpower_db",
            "window_seconds": float(self.topomap_window_seconds),
            "window_samples": int(len(self._session_topomap_buffer)),
            "required_window_samples": int(self.topomap_sample_count),
            "segments": min(segments_by_band.values()) if segments_by_band else 0,
            "segments_by_band": segments_by_band,
            "channels_by_band": channels_by_band or {
                band_name: self._topomap_channel_payload(None) for band_name in TOPO_MAP_BANDS
            },
        }
        if message:
            result["message"] = message
        return result

    def _topomap_channel_payload(self, values: np.ndarray | None) -> list[dict[str, Any]]:
        """把 8 通道 bandpower 数值整理成后端可直接消费的通道列表。"""
        payload = []
        for index, name in enumerate(LIVE_CHANNEL_NAMES):
            x, y = TOPO_MAP_COORDS[name]
            value = None if values is None else float(values[index])
            payload.append({
                "name": name,
                "channel": int(index),
                "protocol_channel": int(DEVICE_STANDARD_MAPPING[name]),
                "value": value,
                "x": float(x),
                "y": float(y),
            })
        return payload

    def _update_session_topomap(self, corrected_uv: np.ndarray) -> None:
        """累计去直流后的数据，每收满一个窗口就分别计算 α/β bandpower。"""
        if self.topomap_sample_count <= 0:
            return
        self._session_topomap_buffer.append(corrected_uv.copy())
        if len(self._session_topomap_buffer) < self.topomap_sample_count:
            return

        window = np.asarray(self._session_topomap_buffer[:self.topomap_sample_count], dtype=np.float64)  # [T, 16]，T=脑地图窗口点数。
        del self._session_topomap_buffer[:self.topomap_sample_count]
        if float(self.sample_rate) <= max(float(band_range[1]) for band_range in TOPO_MAP_BANDS.values()) * 2.0:
            return

        data_8ch = np.stack(
            [window[:, DEVICE_STANDARD_MAPPING[name]] for name in LIVE_CHANNEL_NAMES],
            axis=0,
        )  # [T, 16] -> [8, T]，只取 8 个目标通道并按固定顺序排列。
        for band_name, band_range in TOPO_MAP_BANDS.items():
            powers = []
            for channel_data in data_8ch:
                power = compute_bandpower(
                    channel_data[np.newaxis, np.newaxis, :],
                    float(self.sample_rate),
                    band_range,
                    method=BANDPOWER_METHOD,
                    filter_order=FILTER_ORDER,
                )  # [T] -> [1, 1, 1]，当前通道在当前频段的 bandpower。
                powers.append(float(np.asarray(power, dtype=np.float64).reshape(-1)[0]))
            if all(math.isfinite(power) and power > 0.0 for power in powers):
                self._session_topomap_bandpowers[band_name].append(np.asarray(powers, dtype=np.float64))  # [8]。

    def _render_bandpower_topomap(self, values_db_by_band: dict[str, np.ndarray], image_path: Path) -> None:
        """使用 MNE 绘制 α/β 左右双脑地图，并保存到 profiles。"""
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        import mne

        matplotlib.rcParams["font.sans-serif"] = [
            "Microsoft YaHei",
            "SimHei",
            "SimSun",
            "Arial Unicode MS",
        ]  # 优先使用常见中文字体，保证 PNG 标题里的“α 波 / β 波”能正常显示。
        matplotlib.rcParams["axes.unicode_minus"] = False
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        ch_names = list(LIVE_CHANNEL_NAMES)
        info = mne.create_info(ch_names=ch_names, sfreq=float(self.sample_rate), ch_types=["eeg"] * len(ch_names))
        info.set_montage(mne.channels.make_standard_montage("standard_1020"))

        values_by_band = {
            band_name: np.asarray(values_db_by_band[band_name], dtype=np.float64)
            for band_name in TOPO_MAP_BANDS
        }  # dict[band_name] -> [8]，绘图输入为每个频段 8 个通道的 dB 功率。
        all_values = np.concatenate([values.reshape(-1) for values in values_by_band.values()])
        vmin, vmax = float(np.min(all_values)), float(np.max(all_values))
        if math.isclose(vmin, vmax):
            vmin -= 1.0
            vmax += 1.0

        fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.2))
        last_image = None
        for ax, (band_name, values) in zip(axes, values_by_band.items()):
            last_image, _ = mne.viz.plot_topomap(
                values,
                info,
                axes=ax,
                cmap=plt.get_cmap("jet"),
                cnorm=Normalize(vmin=vmin, vmax=vmax),
                show=False,
                contours=0,
                image_interp="cubic",
                sensors=True,
                outlines="head",
                sphere=None,
            )
            ax.set_title(TOPO_MAP_BAND_TITLES[band_name], fontsize=16, fontweight="normal", pad=12)
        cbar = fig.colorbar(last_image, ax=axes, orientation="vertical", fraction=0.035, pad=0.04)
        cbar.set_label("Bandpower (dB)", fontsize=10)
        cbar.ax.tick_params(labelsize=9)
        fig.subplots_adjust(left=0.04, right=0.88, top=0.88, bottom=0.08, wspace=0.16)
        fig.savefig(image_path, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _live_signal_status_snapshot_unlocked(self) -> dict[str, Any]:
        return {
            key: [
                dict(channel)
                for channel in value
            ] if key == "channels" else value
            for key, value in self._live_quality_status.items()
        }

    @_serialized_feed
    def update_live_signal_quality(self, channels_16: Sequence[object]) -> dict[str, Any]:
        """Update only the live channel-quality path from one 16-channel ADC frame."""
        raw_counts = np.asarray(channels_16, dtype=np.float64)
        if raw_counts.ndim != 1 or raw_counts.size != DEVICE_CHANNEL_COUNT:
            raise ValueError(f"channels_16 must contain exactly {DEVICE_CHANNEL_COUNT} values.")
        if not np.isfinite(raw_counts).all():
            raise ValueError("channels_16 contains NaN or Inf.")
        if np.any(raw_counts < 0.0) or np.any(raw_counts > ADC_COUNT_MAX):
            raise ValueError("channels_16 is outside the unsigned 24-bit ADC-count range.")

        samples = (raw_counts - RAW_ADC_OFFSET) * RAW_ADC_SCALE
        self._update_live_signal_quality(samples)
        return self._live_signal_status_snapshot_unlocked()

    def get_live_signal_status(self) -> dict[str, Any]:
        """返回最近一次实时通道质量评估结果，供后端持续刷新通道状态。"""
        with self._feed_lock:
            return self._live_signal_status_snapshot_unlocked()

    def _make_live_quality_status(
            self,
            status: str,
            ready: bool,
            *,
            channels: list[dict[str, Any]] | None = None,
            message: str | None = None,
    ) -> dict[str, Any]:
        """生成统一格式的实时通道质量状态字典。"""
        if channels is None:
            channels = [
                {
                    "name": name,
                    "channel": index,
                    "protocol_channel": int(DEVICE_STANDARD_MAPPING[name]),
                    "status": status,
                    "status_code": None,
                }
                for index, name in enumerate(LIVE_CHANNEL_NAMES)
            ]
        result = {
            "status": status,
            "ready": bool(ready),
            "window_seconds": float(self.live_quality_window_seconds),
            "window_samples": int(len(self._live_quality_buffer)),
            "required_window_samples": int(self.live_quality_sample_count),
            "step_seconds": float(self.live_quality_step_seconds),
            "filter_warmup_samples": int(self._live_quality_filter_warmup_seen),
            "required_filter_warmup_samples": int(self.live_quality_filter_warmup_sample_count),
            "channels": channels,
        }
        if message:
            result["message"] = message
        return result
    #去直流后的数据，收实时窗口，取 8 通道，50Hz 陷波，8-35Hz 带通，调用 impedance.eeg_signal_impedance()
    #返回 good / acceptable / poor
    def _update_live_signal_quality(self, corrected_uv: np.ndarray) -> None:
        """用去基线后的微伏数据更新滚动窗口，并周期性调用 impedance 评估通道质量。"""

        if self.live_quality_sample_count <= 0:
            self._live_quality_status = self._make_live_quality_status("disabled", False)
            return
        # 8通道、μV、未做,8–30Hz带通的当前EEG数据
        sample_8ch = np.asarray(
            [
                corrected_uv[DEVICE_STANDARD_MAPPING[name]]
                for name in LIVE_CHANNEL_NAMES
            ],
            dtype=np.float64,
        )
        notched, self._live_quality_notch_zi = signal.sosfilt(
            self._live_quality_notch_sos,
            sample_8ch[:, np.newaxis],
            axis=-1,
            zi=self._live_quality_notch_zi,
        )
        filtered, self._live_quality_bandpass_zi = signal.sosfilt(
            self._live_quality_bandpass_sos,
            # sample_8ch[:, np.newaxis],
            notched,
            axis=-1,
            zi=self._live_quality_bandpass_zi,
        )
        filtered_sample_8ch = filtered[:, 0]

        if not self._live_quality_filter_ready:
            self._live_quality_filter_warmup_seen += 1
            self._live_quality_status = self._make_live_quality_status("filter_warming", False)
            if self._live_quality_filter_warmup_seen >= self.live_quality_filter_warmup_sample_count:
                self._live_quality_filter_ready = True
                self._live_quality_buffer.clear()
                self._live_quality_since_update = 0
            return

        # self._live_quality_buffer.append(sample_8ch.copy())
        self._live_quality_buffer.append(filtered_sample_8ch.copy())
        overflow = len(self._live_quality_buffer) - self.live_quality_sample_count
        if overflow > 0:
            del self._live_quality_buffer[:overflow]
        if len(self._live_quality_buffer) < self.live_quality_sample_count:
            self._live_quality_status = self._make_live_quality_status("collecting", False)
            return

        self._live_quality_since_update += 1
        if self._live_quality_status.get("ready") and self._live_quality_since_update < self.live_quality_step_samples:
            return

        impedance_module = _load_impedance_module()
        if impedance_module is None:
            message = _impedance_import_error or "impedance module is not available"
            self._live_quality_status = self._make_live_quality_status("unavailable", False, message=message)
            return

        window = np.asarray(self._live_quality_buffer, dtype=np.float64)  # list[[8]] -> [window_samples, 8]
        data_8ch = window.T  # [window_samples, 8] -> [8, window_samples]
        try:
            quality = impedance_module.eeg_signal_quality(data_8ch, float(self.sample_rate), list(LIVE_CHANNEL_NAMES))
        except Exception as exc:
            self._live_quality_status = self._make_live_quality_status("error", False, message=str(exc))
            return

        channel_eval = quality.get("channel_eval", []) if isinstance(quality, dict) else []
        channels = []
        for index, name in enumerate(LIVE_CHANNEL_NAMES):
            item = channel_eval[index] if index < len(channel_eval) and isinstance(channel_eval[index], dict) else {}
            status_code = item.get("status_code")
            channels.append({
                "name": str(item.get("name", name)),
                "channel": int(item.get("channel", index)),
                "protocol_channel": int(DEVICE_STANDARD_MAPPING[name]),
                "status": str(item.get("status", "unknown")),
                "status_code": None if status_code is None else int(status_code),
            })
        self._live_quality_since_update = 0
        self._live_quality_status = self._make_live_quality_status("ready", True, channels=channels)

    def _filter_live_quality_data(self, data_8ch: np.ndarray) -> np.ndarray:
        """通道质量判断前复用训练/预测预处理滤波：50 Hz 陷波 + 8-30 Hz IIR 带通。"""
        notched = apply_notch_filter(data=data_8ch, sampling_rate=float(self.sample_rate))
        return apply_bandpass_filter(data=notched, sampling_rate=float(self.sample_rate))

    def _apply_prediction_smoothing(self, result: dict[str, Any]) -> dict[str, Any]:
        """Deprecated no-op; formal prediction returns each window directly."""
        return result

    def _apply_prediction_result_latch(self, result: dict[str, Any]) -> dict[str, Any]:
        """Attach legacy display fields while keeping display_code equal to realtime code."""
        if result.get("status") != "predicted":
            return result

        realtime_code = int(result.get("code", 0))
        realtime_label = str(
            result.get("label")
            or result.get("final_result")
            or (ACTION_LABEL if realtime_code == 1 else REST_LABEL)
        )
        if (
                realtime_code == PREDICTION_INVALID_QUALITY
                or result.get("classification_available") is False
        ):
            invalid_result = dict(result)
            invalid_result["realtime_code"] = PREDICTION_INVALID_QUALITY
            invalid_result["realtime_label"] = realtime_label
            invalid_result["display_code"] = PREDICTION_INVALID_QUALITY
            invalid_result["display_label"] = "invalid_quality"
            invalid_result["prediction_result_enabled"] = bool(self._prediction_result_enabled)
            invalid_result["success_latched"] = bool(self._prediction_success_latched)
            invalid_result["prediction_result_latch"] = {
                "enabled": bool(self._prediction_result_enabled),
                "success_latched": bool(self._prediction_success_latched),
                "realtime_code": PREDICTION_INVALID_QUALITY,
                "display_code": PREDICTION_INVALID_QUALITY,
            }
            return invalid_result

        # Success latch is intentionally disabled; legacy fields stay false for compatibility.
        self._prediction_success_latched = False

        display_code = realtime_code

        if display_code == PREDICTION_INVALID_QUALITY:
            display_label = "invalid_quality"
        else:
            display_label = ACTION_LABEL if display_code == 1 else REST_LABEL
        latched_result = dict(result)
        latched_result["realtime_code"] = realtime_code
        latched_result["realtime_label"] = realtime_label
        latched_result["display_code"] = display_code
        latched_result["display_label"] = display_label
        latched_result["prediction_result_enabled"] = bool(self._prediction_result_enabled)
        latched_result["success_latched"] = bool(self._prediction_success_latched)
        latched_result["prediction_result_latch"] = {
            "enabled": bool(self._prediction_result_enabled),
            "success_latched": bool(self._prediction_success_latched),
            "realtime_code": realtime_code,
            "display_code": display_code,
        }
        return latched_result

    @_serialized_feed
    def feed(self, channels_16: Sequence[object]) -> dict[str, Any] | None:
        """接收一帧 16 通道 ADC count，完成基线、质量评估、缓存和算法调用。"""

        raw_counts = np.asarray(channels_16, dtype=np.float64)  # [16], object/int -> [16], float64
        # 检查是不是正好16通道，一维数据，这个错误发生以后，本帧后面的任何算法都不会执行。
        if raw_counts.ndim != 1 or raw_counts.size != DEVICE_CHANNEL_COUNT:
            raise ValueError(f"channels_16 must contain exactly {DEVICE_CHANNEL_COUNT} values.")
        # 检查 NaN / Inf
        if not np.isfinite(raw_counts).all():
            raise ValueError("channels_16 contains NaN or Inf.")
        if np.any(raw_counts < 0.0) or np.any(raw_counts > ADC_COUNT_MAX):
            raise ValueError("channels_16 is outside the unsigned 24-bit ADC-count range.")
        # 数据单位转换，ADC → μV
        samples = (raw_counts - RAW_ADC_OFFSET) * RAW_ADC_SCALE  # [16], ADC count -> [16], microvolts
        raw_uv = samples
        self._trace_adc_seen_samples += 1
        should_trace_adc = (
                self._trace_adc_seen_samples == 1
                or self._trace_adc_seen_samples % self._trace_adc_stride_samples == 0
        )
        if should_trace_adc:
            eeg_trace.trace_adc_to_uv(
                raw_counts,
                samples,
                raw_adc_offset=RAW_ADC_OFFSET,
                raw_adc_scale=RAW_ADC_SCALE,
                raw_uv=raw_uv,
            )
        if self.mode == 1:
            self._update_session_topomap(samples)
        self._update_live_signal_quality(samples)

        # 进入主算法
        adc_like = samples / RAW_ADC_SCALE + RAW_ADC_OFFSET  # [16], microvolts -> [16], ADC-like float
        self._sample_buffer.append(adc_like)
        if len(self._sample_buffer) < self.block_samples:
            return None

        block = np.asarray(self._sample_buffer[:self.block_samples], dtype=np.float64)  # list[[16]] -> [sample_rate, 16]
        del self._sample_buffer[:self.block_samples]

        bridge_block_ready_epoch = time.time()
        bridge_block_ready_perf = time.perf_counter()

        logger.info(
            "[BLOCK READY] "
            "mode=%d time=%.6f "
            "sample_rate=%d block_samples=%d",
            int(self.mode),
            bridge_block_ready_perf,
            int(self.sample_rate),
            int(self.block_samples),
        )

        # profile 已经生成了，但后端还没调用 start_prediction()。
        # 这时继续进来的数据不会再训练，也不会预测，只返回等待状态。
        if self.mode == 0 and self._profile_ready:
            return {
                "success": True,
                "code": 0,
                "status": "profile_ready_waiting_for_prediction",
                "mode": int(self.mode),
                "profile_path": None if self._latest_profile_path is None else str(self._latest_profile_path),
                "message": "Profile is ready; call start_prediction() before prediction blocks are processed.",
            }

        sim_demo_started_epoch = time.time()
        sim_demo_started_perf = time.perf_counter()

        result = sim_demo(
            self.case_no,
            self.age,
            self.gender,
            self.sample_rate,
            self.mode,
            0,
            block,
            DEVICE_STANDARD_MAPPING,
            profile_dir=self.profile_dir,
            required_rest_trials=self.required_rest_trials,
            prediction_step_seconds=self.prediction_step_seconds,
            backend_trial_count=(
                self._backend_trial_count_for_trace
                if self.mode == 1
                else None
            ),
        )

        sim_demo_result_epoch = time.time()
        sim_demo_result_perf = time.perf_counter()

        sim_demo_compute_ms = float((sim_demo_result_perf- sim_demo_started_perf) * 1000.0)
        logger.info("[SIM DEMO COST] " "mode=%d cost_ms=%.3f",int(self.mode),sim_demo_compute_ms,)

        result = dict(result)

        result["bridge_block_ready_epoch"] = float(
            bridge_block_ready_epoch
        )

        result["bridge_block_ready_perf"] = float(
            bridge_block_ready_perf
        )

        result["sim_demo_started_epoch"] = float(
            sim_demo_started_epoch
        )

        result["sim_demo_started_perf"] = float(
            sim_demo_started_perf
        )

        result["sim_demo_result_epoch"] = float(
            sim_demo_result_epoch
        )

        result["sim_demo_result_perf"] = float(
            sim_demo_result_perf
        )

        result["sim_demo_compute_ms"] = float(
            sim_demo_compute_ms
        )



        if result.get("status") == "profile_created":
            self._profile_ready = True
            profile_path = result.get("profile_path")
            self._latest_profile_path = None if profile_path is None else Path(str(profile_path))
            result = dict(result)
            result["mode"] = int(self.mode)
            result["next_action"] = "call start_prediction() to enter prediction mode"
        result = self._apply_prediction_smoothing(result)
        return self._apply_prediction_result_latch(result)
