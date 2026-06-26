"""
增强器基类与注册表
"""
import random
from abc import ABC, abstractmethod
from typing import Dict, Type, Optional


class BaseAugmenter(ABC):
    """所有增强器的抽象基类"""

    def __init__(self, config: dict):
        """
        :param config: 该增强器的配置参数
        """
        self.config = config or {}
        self._initialized = False

    @abstractmethod
    def apply(self, text: str, rng: Optional[random.Random] = None) -> str:
        """
        对单条消息文本应用增强，返回增强后的文本。
        若无变化则返回原文本。

        :param text: 原始文本
        :param rng: 外部随机源；为 None 时使用全局 random，便于复用和复现
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

    @staticmethod
    def _rand(rng: Optional[random.Random] = None):
        """统一随机调用，便于在 rng 为 None 时回退到全局 random"""
        if rng is None:
            return random.random()
        return rng.random()

    @staticmethod
    def _choice(seq, rng: Optional[random.Random] = None):
        if rng is None:
            return random.choice(seq)
        return rng.choice(seq)

    @staticmethod
    def _choices(population, weights=None, rng: Optional[random.Random] = None, k: int = 1):
        if rng is None:
            return random.choices(population, weights=weights, k=k)
        return rng.choices(population, weights=weights, k=k)

    @staticmethod
    def _randint(a: int, b: int, rng: Optional[random.Random] = None) -> int:
        if rng is None:
            return random.randint(a, b)
        return rng.randint(a, b)

    @staticmethod
    def _sample(population, k: int, rng: Optional[random.Random] = None):
        if rng is None:
            return random.sample(population, k)
        return rng.sample(population, k)


class AugmenterRegistry:
    """增强器注册表（支持别名）"""
    _augmenters: Dict[str, Type[BaseAugmenter]] = {}
    _aliases: Dict[str, str] = {}

    @classmethod
    def register(cls, name: str, augmenter_cls: Type[BaseAugmenter], aliases=()):
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
    def list_augmenters(cls):
        return list(cls._augmenters.keys())

    @classmethod
    def list_aliases(cls):
        return dict(cls._aliases)
