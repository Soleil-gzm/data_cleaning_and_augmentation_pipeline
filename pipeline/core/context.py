"""
流水线上下文：薄兼容层，整合配置管理、路径解析、状态追踪

本类作为兼容层，将所有属性和方法转发到具体的管理器类。
新代码应直接使用 ConfigManager、PathResolver、StateTracker。
"""

from pathlib import Path
from typing import Any, Dict, Optional, List
import logging

from .config_manager import ConfigManager
from .path_resolver import PathResolver
from .state_tracker import StateTracker
from ..utils.file_utils import get_file_stats
from ..utils.progress import set_progress_global


class PipelineContext:
    def __init__(self, config: Dict[str, Any]):
        self._config_manager = ConfigManager(config)
        self._resolver = PathResolver(config)
        self._state_tracker = StateTracker(self._resolver.task_dir)
        self._logger = None
        self._step_io_history = []

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

    def clear_step_done(self, step_name: str):
        self._state_tracker.clear_step_done(step_name)

    def resolve_path(self, path_str: str) -> Path:
        return self._resolver.resolve(path_str)

    def get_step_output_dir(self, step_name: str, default_subdir: str = None) -> Path:
        return self._resolver.get_step_output_dir(step_name, default_subdir)

    def get_path(self, path_key: str) -> Path:
        return self._resolver.get_path(path_key)

    def ensure_dir(self, path: Path) -> Path:
        return self._resolver.ensure_dir(path)

    def log_io_summary(
        self, step_name: str, input_paths: List[Path], output_paths: List[Path]
    ):
        if not self.logger:
            return

        total_in_lines = 0
        total_in_size = 0.0
        for p in input_paths:
            stats = get_file_stats(p)
            if stats["exists"]:
                total_in_lines += stats["lines"]
                total_in_size += stats["size_mb"]

        total_out_lines = 0
        total_out_size = 0.0
        for p in output_paths:
            stats = get_file_stats(p)
            if stats["exists"]:
                total_out_lines += stats["lines"]
                total_out_size += stats["size_mb"]

        self.logger.info(f"[{step_name}] IO 转化完成:")
        self.logger.info(
            f"  输入: {len(input_paths)} 个路径, 总计 {total_in_lines} 行, {total_in_size:.2f} MB"
        )
        self.logger.info(
            f"  输出: {len(output_paths)} 个路径, 总计 {total_out_lines} 行, {total_out_size:.2f} MB"
        )
        if total_in_lines > 0:
            ratio = total_out_lines / total_in_lines
            self.logger.info(f"  转化率: {ratio:.2f}x")

        self._step_io_history.append(
            {
                "step": step_name,
                "input_lines": total_in_lines,
                "output_lines": total_out_lines,
                "input_size_mb": total_in_size,
                "output_size_mb": total_out_size,
            }
        )

