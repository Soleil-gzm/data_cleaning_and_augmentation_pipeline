from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict
import logging

from ..core.config_manager import ConfigManager
from ..core.path_resolver import PathResolver


class BaseReporter(ABC):
    """报告器基类"""

    def __init__(self, config: Dict[str, Any], config_manager: ConfigManager, path_resolver: PathResolver):
        self.config = config
        self.config_manager = config_manager
        self.path_resolver = path_resolver
        self.logger = logging.getLogger(self.name)

    @abstractmethod
    def report(self, analysis_data: Dict[str, Any], output_dir: Path, step_name: str):
        """接收分析结果，生成报告"""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__
