"""
结巴模拟增强器（lexical）
- 重复首个中文字符 1~repeat_max 次
"""
from typing import Optional
import re

from ...base import BaseAugmenter
from ....utils.random_utils import randint


class StutterAugmenter(BaseAugmenter):
    def __init__(self, config):
        super().__init__(config)
        self.repeat_max = int(config.get("repeat_max", 2))

    def apply(self, text: str, rng=None) -> str:
        if not isinstance(text, str) or len(text.strip()) == 0:
            return text
        match = re.search(r"[\u4e00-\u9fa5]", text)
        if not match:
            if len(text) > 1:
                return text[0] * 2 + text[1:]
            return text
        char = match.group()
        repeat_count = randint(1, self.repeat_max, rng=rng)
        stuttered = char * (repeat_count + 1)
        start, end = match.start(), match.end()
        return text[:start] + stuttered + text[end:]
