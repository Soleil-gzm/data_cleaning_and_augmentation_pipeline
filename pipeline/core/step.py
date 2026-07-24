"""
步骤抽象基类
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
import logging

from .config_manager import ConfigManager
from .path_resolver import PathResolver
from .state_tracker import StateTracker


class PipelineStep(ABC):
    def __init__(self, config_manager: ConfigManager, path_resolver: PathResolver, state_tracker: StateTracker):
        self.config_manager = config_manager
        self.path_resolver = path_resolver
        self.state_tracker = state_tracker
        self.logger = logging.getLogger(self.name)

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
