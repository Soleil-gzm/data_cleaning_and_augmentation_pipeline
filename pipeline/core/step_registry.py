"""
步骤注册器
"""

from typing import Dict, Type
from .step import PipelineStep
from .context import PipelineContext


class StepRegistry:
    _steps: Dict[str, Type[PipelineStep]] = {}

    @classmethod
    def register(cls, name: str, step_cls: Type[PipelineStep]):
        if not issubclass(step_cls, PipelineStep):
            raise TypeError(f"{step_cls} 不是 PipelineStep 的子类")
        cls._steps[name] = step_cls

    @classmethod
    def get_step(cls, name: str, context: PipelineContext) -> PipelineStep:
        step_cls = cls._steps.get(name)
        if step_cls is None:
            raise ValueError(f"未注册的步骤: {name}")
        return step_cls(context)

    @classmethod
    def list_steps(cls):
        return list(cls._steps.keys())
