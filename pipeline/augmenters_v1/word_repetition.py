"""
词语重复增强器（随机选一个 ≥2 字的中文词重复一次，保持语义自然）
"""
from typing import Optional
import re
import random
import jieba
from .base import BaseAugmenter


class WordRepetitionAugmenter(BaseAugmenter):
    def apply(self, text: str, rng: Optional[random.Random] = None) -> str:
        if not isinstance(text, str) or len(text.strip()) == 0:
            return text

        words = jieba.lcut(text)
        candidates = [
            w for w in words
            if len(w) >= 2 and re.match(r"[\u4e00-\u9fa5]+", w)
        ]
        if not candidates:
            return text

        chosen = self._choice(candidates, rng)
        return text.replace(chosen, chosen + chosen, 1)
