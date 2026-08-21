# eeg_signal_processor.py
import logging
from collections import deque
from typing import Optional, List, Dict, Any

import impedance
import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


class EEGSignalProcessor:
    def __init__(self, sample_rate: int, num_channels: int = 16,
                 stable_duration: float = 0.5, baseline_duration: float = 3.0,
                 window_duration: float = 2.0, selected_channels: Optional[List[int]] = None):
        # 强制窗口至少 1 秒
        if window_duration < 1.0:
            logger.warning(f"传入 window_duration={window_duration}s，强制设为 1.0s")
            window_duration = 1.0
        if baseline_duration < 1.0:
            logger.warning(f"baseline_duration 必须 >=1s，当前 {baseline_duration}s，设为 1.0s")
            baseline_duration = 1.0
        if stable_duration < 0:
            stable_duration = 0.0

        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.stable_duration = stable_duration
        self.baseline_duration = baseline_duration
        self.window_duration = window_duration

        if selected_channels is None:
            self.selected_channels = [1, 4, 7, 8, 9, 12, 11, 10]
        else:
            self.selected_channels = selected_channels

        self.is_baseline_calculated = False
        self.baseline = None
        self.data_buffer = deque(maxlen=int(sample_rate * max(stable_duration, baseline_duration)))
        self.sample_count = 0

        # 窗口样本数至少为 sample_rate（1秒）
        self.window_samples = max(int(sample_rate * window_duration), sample_rate)
        self.filtered_window = deque(maxlen=self.window_samples)
        # logger.info(
        #     f"🔧 EEGSignalProcessor 初始化:\n"
        #     f"   采样率 = {sample_rate} Hz\n"
        #     f"   窗口时长 = {window_duration} s (实际样本数 = {self.window_samples})\n"
        #     f"   基线时长 = {baseline_duration} s\n"
        #     f"   跳过时长 = {stable_duration} s"
        # )

        self._design_filters()
        self.notch_zi = None
        self.bp_zi = None
        self.latest_impedance_status = None

    def _design_filters(self):
        nyquist = self.sample_rate / 2.0
        self.notch_b, self.notch_a = signal.iirnotch(50.0, 30.0, self.sample_rate)
        self.bp_b, self.bp_a = signal.butter(4, [8.0/nyquist, 30.0/nyquist], btype='band')

    def _reset_filter_states(self):
        self.notch_zi = None
        self.bp_zi = None

    def _adc_to_microvolts(self, adc_data: np.ndarray) -> np.ndarray:
        # 两步转换：先减偏移 8388608，再乘 0.0223517
        return (adc_data.astype(np.float64) - 8388608) * 0.0223517

    def _calculate_baseline(self):
        buffer_array = np.array(self.data_buffer)
        self.baseline = np.mean(buffer_array, axis=0)
        logger.info(f"🧬 基线计算完成: {self.baseline.tolist()}")

    def _filter_window(self, window_data: np.ndarray) -> np.ndarray:
        n_samples, n_ch = window_data.shape
        filtered = np.zeros_like(window_data)
        for ch in range(n_ch):
            notch_filtered = signal.filtfilt(self.notch_b, self.notch_a, window_data[:, ch])
            filtered[:, ch] = signal.filtfilt(self.bp_b, self.bp_a, notch_filtered)
        return filtered

    def process_channel_data(self, raw_adc: List[int]) -> Dict[str, Any]:
        if len(raw_adc) != self.num_channels:
            raise ValueError(f"期望{self.num_channels}通道，收到{len(raw_adc)}")

        data = np.array(raw_adc, dtype=np.float64)
        microvolts = self._adc_to_microvolts(data)

        # 跳过不稳定数据
        self.sample_count += 1
        stable_samples = int(self.sample_rate * self.stable_duration)
        if self.sample_count <= stable_samples:
            return {"status": "skipping", "message": f"跳过不稳定数据 {self.sample_count}/{stable_samples}"}

        # 基线计算
        if not self.is_baseline_calculated:
            self.data_buffer.append(microvolts)
            baseline_samples = int(self.sample_rate * self.baseline_duration)
            if len(self.data_buffer) >= baseline_samples:
                self._calculate_baseline()
                self.is_baseline_calculated = True
                self.data_buffer.clear()
                self._reset_filter_states()
                logger.info("✅ 基线就绪，进入实时处理")
                return {"status": "baseline_ready", "message": "基线计算完成"}
            else:
                return {"status": "collecting_baseline",
                        "message": f"收集基线 {len(self.data_buffer)}/{baseline_samples}"}

        # 实时处理
        dc_removed = microvolts - self.baseline
        selected = dc_removed[self.selected_channels]
        self.filtered_window.append(selected)

        current_len = len(self.filtered_window)
        if current_len % 200 == 0:
            logger.debug(f"⏳ 收集窗口: {current_len}/{self.window_samples}")

        if current_len < self.window_samples:
            return {"status": "collecting_window",
                    "message": f"收集窗口 {current_len}/{self.window_samples}"}

        # 窗口已满，进行滤波
        window_array = np.array(self.filtered_window)  # (window_samples, 8)
        logger.debug(f"📊 窗口形状: {window_array.shape}, 样本数: {window_array.shape[0]}")
        filtered_data = self._filter_window(window_array)

        # 二次去直流
        filtered_data = filtered_data - np.mean(filtered_data, axis=0, keepdims=True)

        # ----- 强制检查数据长度 -----
        actual_samples = filtered_data.shape[0]
        logger.debug(f"🔍 滤波后数据样本数: {actual_samples}, 采样率: {self.sample_rate}")
        if actual_samples < self.sample_rate:
            err_msg = (
                f"数据样本数 {actual_samples} 小于采样率 {self.sample_rate}，"
                f"时长 {actual_samples/self.sample_rate:.3f}s，不足 1 秒。"
            )
            logger.error(f"❌ {err_msg}")
            return {"status": "error", "message": err_msg}

        # ----- 调用阻抗评估（关键修改：转置数据） -----
        try:
            # 转置为 (channels, samples) 格式
            filtered_data_transposed = filtered_data.T  # (8, window_samples)
            logger.debug(f"📊 转置后形状: {filtered_data_transposed.shape}")

            impedance_result = impedance.eeg_signal_impedance(filtered_data_transposed, self.sample_rate)
            self.latest_impedance_status = impedance_result

            if impedance_result and 'channel_eval' in impedance_result:
                status_summary = {}
                for ch_info in impedance_result['channel_eval']:
                    status = ch_info.get('status', 'unknown')
                    status_summary[status] = status_summary.get(status, 0) + 1
                logger.debug(f"🧪 阻抗评估完成: {status_summary}")
                if status_summary.get('poor', 0) > 4:
                    logger.warning(f"⚠️ 有 {status_summary.get('poor', 0)} 个通道状态为 poor")

            return {
                "status": "processed",
                "impedance": impedance_result,
                "filtered_data": filtered_data
            }
        except Exception as e:
            logger.error(f"阻抗评估调用失败: {e}")
            return {"status": "error", "message": str(e)}

    # ---------- 查询接口 ----------
    def get_channel_status(self) -> Optional[List[Dict[str, Any]]]:
        if self.latest_impedance_status is None:
            return None
        if isinstance(self.latest_impedance_status, dict) and 'channel_eval' in self.latest_impedance_status:
            return self.latest_impedance_status['channel_eval']
        return None

    def get_channel_status_simple(self) -> Optional[List[str]]:
        channel_eval = self.get_channel_status()
        if channel_eval is None:
            return None
        sorted_list = sorted(channel_eval, key=lambda x: x.get('channel', 0))
        return [item.get('status', 'unknown') for item in sorted_list]

    def update_sample_rate(self, new_rate: int):
        logger.info(f"🔄 更新采样率为 {new_rate}Hz，重置处理器")
        self.sample_rate = new_rate
        self.__init__(
            sample_rate=new_rate,
            num_channels=self.num_channels,
            stable_duration=self.stable_duration,
            baseline_duration=self.baseline_duration,
            window_duration=self.window_duration,
            selected_channels=self.selected_channels
        )