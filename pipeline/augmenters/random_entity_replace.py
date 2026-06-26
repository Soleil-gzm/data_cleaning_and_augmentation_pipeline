"""
随机实体替换（替换数字、日期等）
"""
import re
import random
from .base import BaseAugmenter


class RandomEntityReplaceAugmenter(BaseAugmenter):
    def __init__(self, config):
        super().__init__(config)
        self.prob = config.get("prob", 0.2)

    def apply(self, text: str) -> str:
        def replace_number(match):
            if random.random() < self.prob:
                return str(random.randint(1, 100))
            return match.group()

        def replace_date(match):
            if random.random() < self.prob:
                return f"{random.randint(1,12)}月{random.randint(1,28)}日"
            return match.group()

        text = re.sub(r'\d+', replace_number, text)
        text = re.sub(r'\d+月\d+日', replace_date, text)
        return text