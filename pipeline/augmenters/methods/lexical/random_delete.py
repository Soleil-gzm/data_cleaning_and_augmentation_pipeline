"""
随机删除词增强器（lexical）
"""
from typing import Optional

from ...base import BaseAugmenter
from ...utils import tokenize
from ....utils.random_utils import rand


class RandomDeleteAugmenter(BaseAugmenter):
    def __init__(self, config):
        super().__init__(config)
        self.prob = float(config.get("prob", 0.1))

    def apply(self, text: str, rng=None) -> str:
        if not isinstance(text, str) or not text.strip():
            return text
        tokens = tokenize(text)
        if len(tokens) < 3:
            return text
        new_tokens = [t for t in tokens if rand(rng=rng) > self.prob]
        if not new_tokens or len(new_tokens) < 2:
            return text
        return ''.join(new_tokens)
