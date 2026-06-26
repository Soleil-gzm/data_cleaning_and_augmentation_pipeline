"""
词语重复增强器（随机重复某些词）
"""
import random
from .base import BaseAugmenter
from .utils import tokenize


class WordRepetitionAugmenter(BaseAugmenter):
    def __init__(self, config):
        super().__init__(config)
        self.prob = config.get("prob", 0.15)

    def apply(self, text: str) -> str:
        tokens = tokenize(text)
        if not tokens:
            return text
        new_tokens = []
        for token in tokens:
            new_tokens.append(token)
            if random.random() < self.prob:
                new_tokens.append(token)
        return ''.join(new_tokens)