"""
增强器基类与注册表
================
注册表支持：
    - 按 name 注册
    - 别名（aliases）映射
    - 分类信息统一由 categories.AUGMENTER_META 管理
"""
import random
from abc import ABC, abstractmethod
from typing import Dict, Type, Optional

from .categories import AUGMENTER_META, CATEGORY_LEXICAL


class BaseAugmenter(ABC):
    """所有增强器的抽象基类"""

    def __init__(self, config: dict):
        self.config = config or {}
        self._initialized = False

    @abstractmethod
    def apply(self, text: str, rng: Optional[random.Random] = None) -> str:
        """对单条消息文本应用增强，返回增强后的文本。若无变化则返回原文本。"""
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
    """增强器注册表（支持别名映射，分类信息由 categories 模块管理）"""
    _augmenters: Dict[str, Type[BaseAugmenter]] = {}
    _aliases: Dict[str, str] = {}

    @classmethod
    def register(cls, name: str, augmenter_cls: Type[BaseAugmenter],
                 aliases=(), category: str = None):
        """注册增强器，category 参数已忽略（保留以兼容旧代码）"""
        if not issubclass(augmenter_cls, BaseAugmenter):
            raise TypeError(f"{augmenter_cls} 不是 BaseAugmenter 的子类")
        cls._augmenters[name] = augmenter_cls
        for alias in aliases:
            cls._aliases[alias] = name

    @classmethod
    def get(cls, name: str, config: dict) -> BaseAugmenter:
        real = cls._aliases.get(name, name)
        augmenter_cls = cls._augmenters.get(real)
        if augmenter_cls is None:
            hint = f"（别名指向: {real}）" if real != name else ""
            raise ValueError(f"未注册的增强器: {name}{hint}")
        return augmenter_cls(config)

    @classmethod
    def get_category(cls, name: str) -> str:
        """获取增强器分类，从 categories.AUGMENTER_META 读取"""
        real = cls._aliases.get(name, name)
        meta = AUGMENTER_META.get(real, {})
        return meta.get("category", CATEGORY_LEXICAL)

    @classmethod
    def list_augmenters(cls):
        return list(cls._augmenters.keys())

    @classmethod
    def list_aliases(cls):
        return dict(cls._aliases)

    @classmethod
    def list_by_category(cls):
        out: Dict[str, list] = {}
        for name in cls._augmenters:
            cat = cls.get_category(name)
            out.setdefault(cat, []).append(name)
        return out

    @classmethod
    def clear(cls):
        cls._augmenters.clear()
        cls._aliases.clear()
