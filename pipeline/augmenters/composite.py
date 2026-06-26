"""
组合增强器：支持两种策略
- single: 按权重随机选择一个增强器应用一次（默认）
- multi_step: 按权重抽取 1~N 次叠加应用（更丰富的变体）
"""
import random
from typing import List, Optional
from .base import BaseAugmenter, AugmenterRegistry


class _NullAugmenter(BaseAugmenter):
    def apply(self, text: str, rng=None) -> str:
        return text


class CompositeAugmenter(BaseAugmenter):
    def __init__(self, config: dict):
        super().__init__(config)
        self.augmenters: List[BaseAugmenter] = []
        self.weights: List[float] = []

        augmenters_cfg = config.get("augmenters", {}) if config else {}
        for name, cfg in augmenters_cfg.items():
            if not isinstance(cfg, dict):
                continue
            if not cfg.get("enabled", False):
                continue
            weight = float(cfg.get("weight", 1.0))
            if weight <= 0:
                continue
            aug = AugmenterRegistry.get(name, cfg)
            self.augmenters.append(aug)
            self.weights.append(weight)

        if not self.augmenters:
            self.augmenters.append(_NullAugmenter({}))
            self.weights.append(1.0)

        self.strategy = config.get("strategy", "single") if config else "single"
        default_steps = config.get("default_steps", 2) if config else 2
        self.default_steps = int(default_steps)

    # ---------- 基础能力 ----------
    def _pick_one(self, rng: Optional[random.Random] = None) -> BaseAugmenter:
        if rng is None:
            chosen = random.choices(self.augmenters, weights=self.weights, k=1)[0]
        else:
            chosen = rng.choices(self.augmenters, weights=self.weights, k=1)[0]
        return chosen

    def apply(self, text: str, rng: Optional[random.Random] = None) -> str:
        """按权重选择一个增强器应用一次"""
        chosen = self._pick_one(rng)
        return chosen.apply(text, rng=rng)

    def multi_step_apply(
        self,
        text: str,
        min_steps: int = 1,
        max_steps: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> str:
        """叠加随机选择的增强器多次，产生更丰富的变体"""
        if max_steps is None:
            max_steps = max(min_steps, self.default_steps)
        if max_steps < min_steps:
            max_steps = min_steps

        if rng is None:
            steps = random.randint(min_steps, max_steps)
        else:
            steps = rng.randint(min_steps, max_steps)

        result = text
        for _ in range(steps):
            chosen = self._pick_one(rng)
            result = chosen.apply(result, rng=rng)

        # 若结果未变，重试最多 2 次（避免空变体）
        if result == text and len(text) > 1:
            for _ in range(2):
                cur = text
                for _ in range(steps):
                    chosen = self._pick_one(rng)
                    cur = chosen.apply(cur, rng=rng)
                if cur != text:
                    result = cur
                    break
        return result
