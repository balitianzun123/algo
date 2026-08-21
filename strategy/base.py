# models/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseModelStrategy(ABC):
    """策略基类"""

    @abstractmethod
    def calibration(self, model_type: str,  channels_16: List[int]):
        """处理模型校准"""
        pass

    @abstractmethod
    def prediction(self, model_type: str, channels_16: List[int],) -> Dict[str, Any]:
        """处理模型预测"""
        pass

    @abstractmethod
    def get_channels_status(self,model_type: str) -> Dict[str, Any]:
        """获取脑电通道状态"""
        pass

    @abstractmethod
    def get_algorithm_topmap(self,model_type: str) -> str:
        """获取脑电地形图"""
        pass

    @abstractmethod
    def update_sample_rate(self,model_type: str,rate_hz: int) -> None:
        """设置采样率"""
        pass

    @abstractmethod
    def reset_eeg_params(self,model_type: str) -> None:
        """重置脑电参数"""
        pass

    @abstractmethod
    def update_channels_status(self,model_type: str) -> None:
        """更新通道状态"""
        pass