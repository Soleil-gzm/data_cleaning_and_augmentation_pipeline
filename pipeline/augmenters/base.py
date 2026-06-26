"""
增强器基类与注册表
"""
from abc import ABC, abstractmethod
from typing import Dict, Type, Optional


class BaseAugmenter(ABC):
    """所有增强器的抽象基类"""

    def __init__(self, config: dict):
        """
        :param config: 该增强器的配置参数
        """
        self.config = config
        self._initialized = False

    @abstractmethod
    def apply(self, text: str) -> str:
        """
        对单条消息文本应用增强，返回增强后的文本。
        若无变化则返回原文本。
        """
        pass

    def initialize(self):
        """延迟加载资源（如模型），在首次调用 apply 前执行"""
        if not self._initialized:
            self._load_resources()
            self._initialized = True

    def _load_resources(self):
        """子类重写，加载模型、词典等"""
        pass


class AugmenterRegistry:
    """增强器注册表"""
    _augmenters: Dict[str, Type[BaseAugmenter]] = {}

    @classmethod
    def register(cls, name: str, augmenter_cls: Type[BaseAugmenter]):
        if not issubclass(augmenter_cls, BaseAugmenter):
            raise TypeError(f"{augmenter_cls} 不是 BaseAugmenter 的子类")
        cls._augmenters[name] = augmenter_cls

    @classmethod
    def get(cls, name: str, config: dict) -> BaseAugmenter:
        augmenter_cls = cls._augmenters.get(name)
        if augmenter_cls is None:
            raise ValueError(f"未注册的增强器: {name}")
        return augmenter_cls(config)

    @classmethod
    def list_augmenters(cls):
        return list(cls._augmenters.keys())