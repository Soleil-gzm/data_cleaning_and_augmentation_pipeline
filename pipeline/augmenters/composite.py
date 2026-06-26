"""
组合增强器：根据权重随机选择一种增强方法应用
"""
import random
from typing import List
from .base import BaseAugmenter, AugmenterRegistry


class CompositeAugmenter(BaseAugmenter):
    """
    组合多个增强器，根据权重随机选择其中一个应用。
    每个增强器内部有自己的概率控制是否实际修改文本。
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.augmenters: List[BaseAugmenter] = []
        self.weights: List[float] = []

        augmenters_cfg = config.get("augmenters", {})
        for name, cfg in augmenters_cfg.items():
            if cfg.get("enabled", False):
                weight = cfg.get("weight", 1.0)
                if weight > 0:
                    aug = AugmenterRegistry.get(name, cfg)
                    self.augmenters.append(aug)
                    self.weights.append(weight)

        if not self.augmenters:
            # 如果没有启用的增强器，使用空增强器
            self.augmenters.append(_NullAugmenter({}))
            self.weights.append(1.0)

    def apply(self, text: str) -> str:
        # 按权重选择一个增强器
        chosen = random.choices(self.augmenters, weights=self.weights)[0]
        return chosen.apply(text)


class _NullAugmenter(BaseAugmenter):
    """空增强器，直接返回原文本"""
    def apply(self, text: str) -> str:
        return text