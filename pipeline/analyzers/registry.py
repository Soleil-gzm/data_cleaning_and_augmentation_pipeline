from typing import Dict, Type
from .base import BaseAnalyzer


class AnalyzerRegistry:
    _analyzers: Dict[str, Type[BaseAnalyzer]] = {}

    @classmethod
    def register(cls, name: str, analyzer_cls: Type[BaseAnalyzer]):
        cls._analyzers[name] = analyzer_cls

    @classmethod
    def get_analyzer(cls, name: str, context) -> BaseAnalyzer:
        return cls._analyzers[name](context)

    @classmethod
    def list_analyzers(cls):
        return list(cls._analyzers.keys())