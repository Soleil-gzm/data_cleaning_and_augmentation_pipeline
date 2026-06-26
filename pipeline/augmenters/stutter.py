"""
结巴模拟增强器（在词间插入停顿）
"""
import random
from .base import BaseAugmenter
from .utils import tokenize


class StutterAugmenter(BaseAugmenter):
    def __init__(self, config):
        super().__init__(config)
        self.prob = config.get("prob", 0.2)

    def apply(self, text: str) -> str:
        tokens = tokenize(text)
        if len(tokens) < 2:
            return text
        new_tokens = []
        for token in tokens:
            new_tokens.append(token)
            if random.random() < self.prob and len(token) > 1:
                # 重复第一个字并加省略号
                new_tokens.append(token[0] + '...')
        return ''.join(new_tokens)