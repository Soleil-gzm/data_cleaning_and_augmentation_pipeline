from typing import Dict, Type, Any
from .base import BaseReporter


class ReporterRegistry:
    _reporters: Dict[str, Type[BaseReporter]] = {}

    @classmethod
    def register(cls, name: str, reporter_cls: Type[BaseReporter]):
        cls._reporters[name] = reporter_cls

    @classmethod
    def get_reporter(cls, name: str, config: Dict[str, Any], context) -> BaseReporter:
        return cls._reporters[name](config, context)

    @classmethod
    def list_reporters(cls):
        return list(cls._reporters.keys())