from typing import Dict, Type
from .base import BaseAnalyzer
from ..core.config_manager import ConfigManager
from ..core.path_resolver import PathResolver


class AnalyzerRegistry:
    _analyzers: Dict[str, Type[BaseAnalyzer]] = {}

    @classmethod
    def register(cls, name: str, analyzer_cls: Type[BaseAnalyzer]):
        if not issubclass(analyzer_cls, BaseAnalyzer):
            raise TypeError(f"{analyzer_cls} 不是 BaseAnalyzer 的子类")
        cls._analyzers[name] = analyzer_cls

    @classmethod
    def get_analyzer(cls, name: str, config_manager: ConfigManager, path_resolver: PathResolver) -> BaseAnalyzer:
        analyzer_cls = cls._analyzers.get(name)
        if analyzer_cls is None:
            raise ValueError(f"未注册的分析器: {name}")
        return analyzer_cls(config_manager, path_resolver)

    @classmethod
    def list_analyzers(cls):
        return list(cls._analyzers.keys())
