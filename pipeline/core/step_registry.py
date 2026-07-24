"""
步骤注册器
"""

from typing import Dict, Type
from .step import PipelineStep
from .config_manager import ConfigManager
from .path_resolver import PathResolver
from .state_tracker import StateTracker


class StepRegistry:
    _steps: Dict[str, Type[PipelineStep]] = {}

    @classmethod
    def register(cls, name: str, step_cls: Type[PipelineStep]):
        if not issubclass(step_cls, PipelineStep):
            raise TypeError(f"{step_cls} 不是 PipelineStep 的子类")
        cls._steps[name] = step_cls

    @classmethod
    def get_step(cls, name: str, config_manager: ConfigManager, path_resolver: PathResolver, state_tracker: StateTracker) -> PipelineStep:
        step_cls = cls._steps.get(name)
        if step_cls is None:
            raise ValueError(f"未注册的步骤: {name}")
        return step_cls(config_manager, path_resolver, state_tracker)

    @classmethod
    def list_steps(cls):
        return list(cls._steps.keys())
