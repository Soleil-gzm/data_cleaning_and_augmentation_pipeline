"""
流水线上下文：管理配置、路径、日志、断点、IO审计
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, List
import logging

from ..utils.file_utils import get_file_stats, print_directory_tree
from ..utils.progress import set_progress_global


class PipelineContext:
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._task_name = config.get("task_name", "default_task")
        self._intermediate_root = Path(config.get("paths", {}).get("intermediate", "./intermediate"))
        self._output_root = Path(config.get("paths", {}).get("output", "./output"))
        self._task_dir = self._intermediate_root / self._task_name
        self._resume = config.get("resume", False)
        self._logger = None
        self._step_io_history = []  # 记录每步IO

        # 进度条全局开关
        show_progress = config.get("logging", {}).get("show_progress", True)
        set_progress_global(show_progress)

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @property
    def task_name(self) -> str:
        return self._task_name

    @property
    def task_dir(self) -> Path:
        return self._task_dir

    @property
    def intermediate_root(self) -> Path:
        return self._intermediate_root

    @property
    def output_root(self) -> Path:
        return self._output_root

    @property
    def resume(self) -> bool:
        return self._resume

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            return logging.getLogger("PipelineContext")
        return self._logger

    def set_logger(self, logger: logging.Logger):
        self._logger = logger

    def get_step_config(self, step_name: str) -> Dict[str, Any]:
        steps = self._config.get("steps", {})
        return steps.get(step_name, {})

    def is_step_enabled(self, step_name: str) -> bool:
        step_cfg = self.get_step_config(step_name)
        return step_cfg.get("enabled", True)

    def is_step_done(self, step_name: str) -> bool:
        flag_path = self.task_dir / f".step_{step_name}_done"
        return flag_path.exists()

    def mark_step_done(self, step_name: str):
        flag_path = self.task_dir / f".step_{step_name}_done"
        flag_path.touch()

    def clear_step_done(self, step_name: str):
        flag_path = self.task_dir / f".step_{step_name}_done"
        if flag_path.exists():
            flag_path.unlink()

    def resolve_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        return self.task_dir / p

    def get_step_output_dir(self, step_name: str, default_subdir: str = None) -> Path:
        step_cfg = self.get_step_config(step_name)
        out_dir = step_cfg.get("output_dir")
        if out_dir:
            return self.resolve_path(out_dir)
        if default_subdir:
            return self.task_dir / default_subdir
        return self.task_dir / step_name

    # ===== IO 审计 =====
    def log_io_summary(self, step_name: str, input_paths: List[Path], output_paths: List[Path]):
        """记录输入输出转化统计"""
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
        self.logger.info(f"  输入: {len(input_paths)} 个路径, 总计 {total_in_lines} 行, {total_in_size:.2f} MB")
        self.logger.info(f"  输出: {len(output_paths)} 个路径, 总计 {total_out_lines} 行, {total_out_size:.2f} MB")
        if total_in_lines > 0:
            ratio = total_out_lines / total_in_lines
            self.logger.info(f"  转化率: {ratio:.2f}x")

        self._step_io_history.append({
            "step": step_name,
            "input_lines": total_in_lines,
            "output_lines": total_out_lines,
            "input_size_mb": total_in_size,
            "output_size_mb": total_out_size,
        })

    def print_task_tree(self):
        """打印任务目录树"""
        if not self._config.get("logging", {}).get("print_tree", True):
            return
        if self.logger:
            self.logger.info(f"任务目录结构 ({self.task_dir}):")
        print_directory_tree(self.task_dir, max_depth=4, exclude_patterns=[".step_*", "__pycache__", "*.pyc", ".DS_Store", "*.log"])