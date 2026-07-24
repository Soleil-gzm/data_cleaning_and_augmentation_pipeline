"""
词语重复增强器（lexical）
- 随机选一个 ≥2 字的中文词重复一次，保持语义自然
"""
from typing import Optional
import re

from ...base import BaseAugmenter
from ....utils.random_utils import choice


class WordRepetitionAugmenter(BaseAugmenter):
    def apply(self, text: str, rng=None) -> str:
        if not isinstance(text, str) or not text.strip():
            return text

        try:
            import jieba
        except ImportError:
            return text

        words = list(jieba.lcut(text))
        candidates = [
            w for w in words
            if len(w) >= 2 and re.match(r"[\u4e00-\u9fa5]+", w)
        ]
        if not candidates:
            return text

        chosen = choice(candidates, rng=rng)
        return text.replace(chosen, chosen + chosen, 1)
