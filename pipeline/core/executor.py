"""
执行器基类：预留并行扩展接口
"""
from abc import ABC, abstractmethod
from typing import List, Callable, Any
from .context import PipelineContext


class BaseExecutor(ABC):
    def __init__(self, context: PipelineContext):
        self.context = context

    @abstractmethod
    def execute(self, tasks: List[Callable[[], bool]]) -> List[bool]:
        """执行任务列表，返回每个任务的成功状态"""
        pass


class SequentialExecutor(BaseExecutor):
    """顺序执行器（当前实现）"""

    def execute(self, tasks: List[Callable[[], bool]]) -> List[bool]:
        results = []
        for i, task in enumerate(tasks):
            results.append(task())
        return results