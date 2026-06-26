"""
语序打乱增强器（随机交换相邻词或短语）
"""
import random
from .base import BaseAugmenter
from .utils import tokenize


class ReorderAugmenter(BaseAugmenter):
    def __init__(self, config):
        super().__init__(config)
        self.prob = config.get("prob", 0.2)

    def apply(self, text: str) -> str:
        tokens = tokenize(text)
        if len(tokens) < 3:
            return text
        new_tokens = tokens[:]
        for i in range(len(new_tokens) - 1):
            if random.random() < self.prob:
                new_tokens[i], new_tokens[i+1] = new_tokens[i+1], new_tokens[i]
        return ''.join(new_tokens)