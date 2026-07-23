"""
流水线上下文：薄兼容层，整合配置管理、路径解析、状态追踪

本类作为兼容层，将所有属性和方法转发到具体的管理器类。
新代码应直接使用 ConfigManager、PathResolver、StateTracker。
"""

from pathlib import Path
from typing import Any, Dict, Optional
import logging

from .config_manager import ConfigManager
from .path_resolver import PathResolver
from .state_tracker import StateTracker
from ..utils.progress import set_progress_global


class PipelineContext:
    def __init__(self, config: Dict[str, Any]):
        self._config_manager = ConfigManager(config)
        self._resolver = PathResolver(config)
        self._state_tracker = StateTracker(self._resolver.task_dir)
        self._logger = None

        show_progress = config.get("logging", {}).get("show_progress", True)
        set_progress_global(show_progress)

    @property
    def config(self) -> Dict[str, Any]:
        return self._config_manager.config

    @property
    def task_name(self) -> str:
        return self._config_manager.task_name

    @property
    def task_dir(self) -> Path:
        return self._resolver.task_dir

    @property
    def intermediate_root(self) -> Path:
        return self._resolver.intermediate_root

    @property
    def output_root(self) -> Path:
        return self._resolver.output_root

    @property
    def resume(self) -> bool:
        return self._config_manager.resume

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            return logging.getLogger("PipelineContext")
        return self._logger

    def set_logger(self, logger: logging.Logger):
        self._logger = logger

    def get_step_config(self, step_name: str) -> Dict[str, Any]:
        return self._config_manager.get_step_config(step_name)

    def is_step_enabled(self, step_name: str) -> bool:
        return self._config_manager.is_step_enabled(step_name)

    def is_step_done(self, step_name: str) -> bool:
        return self._state_tracker.is_step_done(step_name)

    def mark_step_done(self, step_name: str):
        self._state_tracker.mark_step_done(step_name)

    def resolve_path(self, path_str: str) -> Path:
        return self._resolver.resolve(path_str)

    def get_step_output_dir(self, step_name: str, default_subdir: str = None) -> Path:
        return self._resolver.get_step_output_dir(step_name, default_subdir)

    def get_path(self, path_key: str) -> Path:
        return self._resolver.get_path(path_key)

    def ensure_dir(self, path: Path) -> Path:
        return self._resolver.ensure_dir(path)

    def get_input_file(self, file_key: str) -> Optional[Path]:
        return self._resolver.get_input_file(file_key)