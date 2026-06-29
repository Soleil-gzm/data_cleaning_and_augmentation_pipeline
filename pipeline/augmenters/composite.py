"""
组合增强器
==========
- 支持按 category 路由（lexical / order / model）
- 支持 single / multi_step 两种策略
- single 策略内置 fallback：若首次选择的增强器未产生变化，
  会从同类别剩余增强器中再挑重试最多 retry_times 次
- 支持通过 "enabled_categories" 限制启用的增强器类别
  （例如只启用 lexical、order 两类，不加载 model）
"""
import random
from typing import List, Optional, Dict

from .base import BaseAugmenter, AugmenterRegistry
from .categories import CATEGORY_LEXICAL, CATEGORY_ORDER, CATEGORY_MODEL


class _NullAugmenter(BaseAugmenter):
    def apply(self, text: str, rng=None) -> str:
        return text


class CompositeAugmenter(BaseAugmenter):
    def __init__(self, config: dict):
        super().__init__(config)
        self.augmenters: List[BaseAugmenter] = []
        self.weights: List[float] = []
        self.by_category: Dict[str, List[int]] = {
            CATEGORY_LEXICAL: [],
            CATEGORY_ORDER: [],
            CATEGORY_MODEL: [],
        }
        self.names: List[str] = []

        cfg = config or {}
        augmenters_cfg = cfg.get("augmenters", {}) if cfg else {}
        enabled_cats = cfg.get("enabled_categories", None)  # None = 全启用
        self._enabled_categories = enabled_cats

        for name, sub in augmenters_cfg.items():
            if not isinstance(sub, dict):
                continue
            if not sub.get("enabled", False):
                continue
            cat = AugmenterRegistry.get_category(name) or CATEGORY_LEXICAL
            if enabled_cats is not None and cat not in enabled_cats:
                continue
            weight = float(sub.get("weight", 1.0))
            if weight <= 0:
                continue
            aug = AugmenterRegistry.get(name, sub)
            self.augmenters.append(aug)
            self.weights.append(weight)
            self.names.append(name)
            self.by_category.setdefault(cat, []).append(len(self.augmenters) - 1)

        if not self.augmenters:
            self.augmenters.append(_NullAugmenter({}))
            self.weights.append(1.0)
            self.names.append("__null__")

        self.strategy = cfg.get("strategy", "single") if cfg else "single"
        default_steps = cfg.get("default_steps", 2) if cfg else 2
        self.default_steps = int(default_steps)
        self.single_retry = int(cfg.get("single_retry", 3))
        self.multi_retry = int(cfg.get("multi_retry", 2))

    # ---------- 基础能力 ----------
    def _pick_one(self, rng: Optional[random.Random] = None,
                 category: Optional[str] = None) -> BaseAugmenter:
        if category is None or category not in self.by_category or not self.by_category[category]:
            candidates_idx = list(range(len(self.augmenters)))
            weights = self.weights
        else:
            candidates_idx = self.by_category[category]
            weights = [self.weights[i] for i in candidates_idx]

        if not candidates_idx:
            return self.augmenters[0]

        if rng is None:
            chosen_idx = random.choices(candidates_idx, weights=weights, k=1)[0]
        else:
            chosen_idx = rng.choices(candidates_idx, weights=weights, k=1)[0]
        return self.augmenters[chosen_idx]

    def _distinct_pick(self, tried: set, rng, category=None):
        """挑一个未尝试过的增强器；全部试过则回退到常规 pick"""
        if category is None or category not in self.by_category or not self.by_category[category]:
            pool = list(range(len(self.augmenters)))
        else:
            pool = list(self.by_category[category])
        candidates = [i for i in pool if i not in tried]
        if not candidates:
            return self._pick_one(rng, category)
        weights = [self.weights[i] for i in candidates]
        if rng is None:
            idx = random.choices(candidates, weights=weights, k=1)[0]
        else:
            idx = rng.choices(candidates, weights=weights, k=1)[0]
        tried.add(idx)
        return self.augmenters[idx]

    # ---------- 对外 API ----------
    def apply(self, text: str, rng: Optional[random.Random] = None) -> str:
        """按权重选择一个增强器应用一次；带 fallback"""
        tried: set = set()
        last_result = text
        for attempt in range(max(1, self.single_retry)):
            chosen = self._distinct_pick(tried, rng)
            last_result = chosen.apply(text, rng=rng)
            if last_result != text:
                return last_result
            if len(tried) >= len(self.augmenters):
                break
        return last_result

    def multi_step_apply(
        self,
        text: str,
        min_steps: int = 1,
        max_steps: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> str:
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

        if result == text and len(text) > 1:
            for _ in range(self.multi_retry):
                cur = text
                for _ in range(steps):
                    chosen = self._pick_one(rng)
                    cur = chosen.apply(cur, rng=rng)
                if cur != text:
                    result = cur
                    break
        return result

    # ---------- 统计 ----------
    def enabled_names(self) -> List[str]:
        return [n for n in self.names if n != "__null__"]

    def enabled_categories(self):
        out: List[str] = []
        for cat, idxs in self.by_category.items():
            if idxs:
                out.append(cat)
        return out
