"""
报告器注册表
"""

from typing import Dict, Type, Any
from .base import BaseReporter


class ReporterRegistry:
    _reporters: Dict[str, Type[BaseReporter]] = {}

    @classmethod
    def register(cls, name: str, reporter_cls: Type[BaseReporter]):
        if not issubclass(reporter_cls, BaseReporter):
            raise TypeError(f"{reporter_cls} 不是 BaseReporter 的子类")
        cls._reporters[name] = reporter_cls

    @classmethod
    def get_reporter(cls, name: str, config: Dict[str, Any], context) -> BaseReporter:
        reporter_cls = cls._reporters.get(name)
        if reporter_cls is None:
            raise ValueError(f"未注册的报告器: {name}")
        return reporter_cls(config, context)

    @classmethod
    def list_reporters(cls):
        return list(cls._reporters.keys())
