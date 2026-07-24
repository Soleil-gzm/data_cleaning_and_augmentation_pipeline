from abc import ABC, abstractmethod
from typing import Any, Dict
import logging

from ..core.config_manager import ConfigManager
from ..core.path_resolver import PathResolver


class BaseAnalyzer(ABC):
    """分析器基类"""

    def __init__(self, config_manager: ConfigManager, path_resolver: PathResolver):
        self.config_manager = config_manager
        self.path_resolver = path_resolver
        self.logger = logging.getLogger(self.name)

    @abstractmethod
    def analyze(self, raw_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """接收原始指标，返回分析结果"""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__
