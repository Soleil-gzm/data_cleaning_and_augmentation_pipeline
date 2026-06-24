from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class BaseReporter(ABC):
    """报告器基类"""

    def __init__(self, config: Dict[str, Any], context):
        self.config = config
        self.context = context
        self.logger = context.logger

    @abstractmethod
    def report(self, analysis_data: Dict[str, Any], output_dir: Path, step_name: str):
        """接收分析结果，生成报告"""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__
