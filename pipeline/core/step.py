"""
步骤抽象基类
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from .context import PipelineContext


class PipelineStep(ABC):
    def __init__(self, context: PipelineContext):
        self.context = context
        self.logger = context.logger

    @abstractmethod
    def run(self) -> bool:
        """执行核心逻辑"""
        pass

    def pre_run(self) -> bool:
        """前置钩子"""
        return True

    def post_run(self) -> bool:
        """后置钩子（可重写以添加IO审计）"""
        return True

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def _get_input_paths(self) -> List[Path]:
        """子类重写，返回输入路径列表供审计"""
        return []

    def _get_output_paths(self) -> List[Path]:
        """子类重写，返回输出路径列表供审计"""
        return []
