"""
流水线调度器
- 步骤按顺序执行
- 步骤内部的并行由各步骤自己控制（通过 max_workers 参数）
"""

import sys
from pathlib import Path
from typing import Optional

from .config_manager import ConfigManager
from .path_resolver import PathResolver
from .state_tracker import StateTracker
from .step_registry import StepRegistry
from ..config.loader import ConfigLoader
from ..utils.logger import setup_task_logger
from ..utils.progress import set_progress_global


class Pipeline:
    def __init__(
        self, config_path: Optional[Path] = None, config_dict: Optional[dict] = None
    ):
        if config_path is not None and config_dict is not None:
            raise ValueError("只能指定 config_path 或 config_dict 之一")
        if config_path is not None:
            self.config = ConfigLoader.load(config_path)
        elif config_dict is not None:
            self.config = config_dict
        else:
            raise ValueError("必须提供 config_path 或 config_dict")

        # 初始化服务
        self._config_manager = ConfigManager(self.config)
        self._path_resolver = PathResolver(self.config)
        self._state_tracker = StateTracker(self._path_resolver.task_dir)

        # 进度条设置
        show_progress = self.config.get("logging", {}).get("show_progress", True)
        set_progress_global(show_progress)

        # 设置日志
        task_name = self._config_manager.task_name
        task_dir = self._path_resolver.task_dir
        log_cfg = self.config.get("logging", {})
        logger = setup_task_logger(
            task_name,
            task_dir / "logs",
            console_level=log_cfg.get("level", "INFO"),
            file_level=log_cfg.get("file_level", "DEBUG"),
        )
        self.logger = logger

        # 步骤执行顺序
        self.steps_order = self.config.get("steps_order", [])
        if not self.steps_order:
            default_order = [
                "01_split",
                "02_bucket",
                "03_clean",
                "05_augment",
            ]
            self.steps_order = [
                s for s in default_order if s in self.config.get("steps", {})
            ]



    def run(self, step_name: Optional[str] = None) -> bool:
        if step_name:
            return self._run_single(step_name)

        results = []
        for name in self.steps_order:
            results.append(self._run_single(name))

        if all(results):
            self.logger.info("✅ 所有步骤执行完毕！")
            return True
        else:
            failed_indices = [i for i, r in enumerate(results) if not r]
            for i in failed_indices:
                self.logger.error(f"步骤 {self.steps_order[i]} 失败")
            return False

    def _run_single(self, name: str) -> bool:
        if name not in self.config.get("steps", {}):
            self.logger.warning(f"配置中未定义步骤 {name}，跳过")
            return True

        if not self._config_manager.is_step_enabled(name):
            self.logger.info(f"步骤 {name} 已禁用，跳过")
            return True

        if self._config_manager.resume and self._state_tracker.is_step_done(name):
            self.logger.info(f"步骤 {name} 已完成（断点续跑），跳过")
            return True

        self.logger.info(f"🚀 开始执行步骤: {name}")

        try:
            step = StepRegistry.get_step(name, self._config_manager, self._path_resolver, self._state_tracker)
        except ValueError as e:
            self.logger.error(f"步骤 {name} 未注册: {e}")
            return False

        if not step.pre_run():
            self.logger.warning(f"步骤 {name} 前置钩子返回 False，跳过")
            return True

        try:
            success = step.run()
        except Exception as e:
            self.logger.exception(f"步骤 {name} 执行异常: {e}")
            success = False

        if success:
            try:
                step.post_run()
            except Exception as e:
                self.logger.exception(f"步骤 {name} 后置钩子异常: {e}")
                success = False

        if success:
            self._state_tracker.mark_step_done(name)
            self.logger.info(f"✅ 步骤 {name} 执行成功")
        else:
            self.logger.error(f"❌ 步骤 {name} 执行失败")

        return success
