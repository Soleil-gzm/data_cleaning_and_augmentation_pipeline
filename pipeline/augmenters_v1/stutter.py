"""
结巴模拟增强器（重复首个中文字符 1~2 次，更自然）
"""
from typing import Optional
import re
import random
from .base import BaseAugmenter


class StutterAugmenter(BaseAugmenter):
    def __init__(self, config):
        super().__init__(config)
        self.repeat_max = int(config.get("repeat_max", 2))

    def apply(self, text: str, rng: Optional[random.Random] = None) -> str:
        if not isinstance(text, str) or len(text.strip()) == 0:
            return text
        match = re.search(r"[\u4e00-\u9fa5]", text)
        if not match:
            if len(text) > 1:
                return text[0] * 2 + text[1:]
            return text
        char = match.group()
        repeat_count = self._randint(1, self.repeat_max, rng)
        stuttered = char * (repeat_count + 1)
        start, end = match.start(), match.end()
        return text[:start] + stuttered + text[end:]
