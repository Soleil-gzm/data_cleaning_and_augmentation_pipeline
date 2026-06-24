"""
流水线调度器
"""

import sys
from pathlib import Path
from typing import Optional
from .context import PipelineContext
from .executor import SequentialExecutor, BaseExecutor
from .step_registry import StepRegistry
from ..config.loader import ConfigLoader
from ..utils.logger import setup_task_logger


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

        self.context = PipelineContext(self.config)

        # 设置日志
        task_name = self.context.task_name
        task_dir = self.context.task_dir
        log_cfg = self.config.get("logging", {})
        logger = setup_task_logger(
            task_name,
            task_dir / "logs",
            console_level=log_cfg.get("level", "INFO"),
            file_level=log_cfg.get("file_level", "DEBUG"),
        )
        self.context.set_logger(logger)
        self.logger = logger

        # 打印目录树
        self.context.print_task_tree()

        # 步骤顺序
        self.steps_order = self.config.get("steps_order", [])
        if not self.steps_order:
            default_order = [
                "00_generate_raw",
                "01_split",
                "02_bucket",
                "03_clean",
                "04_finalize",
                "05_augment",
                "06_replace_text",
            ]
            self.steps_order = [
                s for s in default_order if s in self.config.get("steps", {})
            ]

        # 执行器
        executor_type = self.config.get("executor", {}).get("type", "sequential")
        if executor_type == "sequential":
            self.executor = SequentialExecutor(self.context)
        else:
            self.logger.warning(f"执行器 {executor_type} 未实现，降级为 sequential")
            self.executor = SequentialExecutor(self.context)

    def run(self, step_name: Optional[str] = None) -> bool:
        if step_name:
            return self._run_single(step_name)

        # 顺序执行
        tasks = [lambda n=name: self._run_single(n) for name in self.steps_order]
        results = self.executor.execute(tasks)

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

        if not self.context.is_step_enabled(name):
            self.logger.info(f"步骤 {name} 已禁用，跳过")
            return True

        if self.context.resume and self.context.is_step_done(name):
            self.logger.info(f"步骤 {name} 已完成（断点续跑），跳过")
            return True

        self.logger.info(f"🚀 开始执行步骤: {name}")

        try:
            step = StepRegistry.get_step(name, self.context)
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
            self.context.mark_step_done(name)
            self.logger.info(f"✅ 步骤 {name} 执行成功")
        else:
            self.logger.error(f"❌ 步骤 {name} 执行失败")

        return success
