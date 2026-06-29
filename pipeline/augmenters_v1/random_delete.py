"""
随机删除词增强器
"""
from typing import Optional
import random
from .base import BaseAugmenter
from .utils import tokenize


class RandomDeleteAugmenter(BaseAugmenter):
    def __init__(self, config):
        super().__init__(config)
        self.prob = config.get("prob", 0.1)

    def apply(self, text: str, rng: Optional[random.Random] = None) -> str:
        tokens = tokenize(text)
        if len(tokens) < 3:
            return text
        new_tokens = [t for t in tokens if self._rand(rng) > self.prob]
        if not new_tokens:
            return text
        return ''.join(new_tokens)
