from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAnalyzer(ABC):
    """分析器基类"""
    def __init__(self, context):
        self.context = context
        self.logger = context.logger

    @abstractmethod
    def analyze(self, raw_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """接收原始指标，返回分析结果"""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__