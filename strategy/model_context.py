# models/model_context.py
import logging
from typing import List, Any

from .data_model import DataModelStrategy

logger = logging.getLogger(__name__)


class ModelStrategyContext:
    """策略上下文"""

    def __init__(self):
        # 模块级 model_context 持有一个长期存在的 DataModelStrategy 实例
        self._strategies = {
            0: DataModelStrategy(),  # 第一次调用会创建实例，后续调用返回同一个实例
            # "ml": MachineLearningModelStrategy(),
            # "dl": DeepLearningModelStrategy(),
        }
        logger.info(f"已注册策略: {list(self._strategies.keys())}")
        # 验证单例：多次获取应该返回相同对象
        # instance1 = DataModelStrategy()
        # instance2 = DataModelStrategy()
        # logger.info(f"DataModelStrategy 单例验证: id1={id(instance1)}, id2={id(instance2)}, 是否相同={instance1 is instance2}")

    def calibration(self, model_type: int, channels_16: List[int]) -> None:
        if model_type not in self._strategies:
            raise ValueError(f"不支持的模型类型: {model_type}")
        return self._strategies[model_type].calibration(channels_16)

    def prediction(self, model_type: int, channels_16: List[int]) -> int:
        if model_type not in self._strategies:
            raise ValueError(f"不支持的模型类型: {model_type}")
        return self._strategies[model_type].prediction(channels_16)

    def get_channels_status(self, model_type: int) -> dict[str, Any]:
        if model_type not in self._strategies:
            raise ValueError(f"不支持的模型类型: {model_type}")
        return self._strategies[model_type].get_channels_status()

    def get_calibration_accepted_trials(self, model_type: int) -> int:
        if model_type not in self._strategies:
            raise ValueError(f"不支持的模型类型: {model_type}")
        return self._strategies[model_type].get_calibration_accepted_trials()

    def get_calibration_profile_ready(self,model_type: int,) -> bool:
        if model_type not in self._strategies:
            raise ValueError( f"不支持的模型类型: {model_type}")
        return self._strategies[model_type].get_calibration_profile_ready()

    def update_channels_status(self, model_type: int, channels_16: List[int]) -> dict[str, Any]:
        if model_type not in self._strategies:
            raise ValueError(f"涓嶆敮鎸佺殑妯″瀷绫诲瀷: {model_type}")
        return self._strategies[model_type].update_channels_status(channels_16)

    def get_algorithm_topmap(self, model_type: int) -> str:
        if model_type not in self._strategies:
            raise ValueError(f"不支持的模型类型: {model_type}")
        return self._strategies[model_type].get_algorithm_topmap()

    def update_sample_rate(self, model_type: int, rate_hz: int) -> None:
        if model_type not in self._strategies:
            raise ValueError(f"不支持的模型类型: {model_type}")
        return self._strategies[model_type].update_sample_rate(rate_hz)

    def reset_eeg_params(self, model_type: int) -> None:
        if model_type not in self._strategies:
            raise ValueError(f"不支持的模型类型: {model_type}")
        return self._strategies[model_type].reset_eeg_params()

model_context = ModelStrategyContext()
