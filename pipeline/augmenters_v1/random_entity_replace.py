"""
随机实体替换增强器
- 使用 bank.txt 词库替换机构/实体名
- 数字/日期替换带长度和有效性约束（避免产生"2月30日"、卡号变化过大）
"""
from typing import Optional
import re
import random
from pathlib import Path
from .base import BaseAugmenter
from .utils import load_word_set


class RandomEntityReplaceAugmenter(BaseAugmenter):
    def _load_resources(self):
        dict_path = self.config.get("dict_path", "resources/bank.txt")
        self.entity_words = sorted(load_word_set(dict_path))
        self.prob = float(self.config.get("prob", 0.2))

    def apply(self, text: str, rng: Optional[random.Random] = None) -> str:
        self.initialize()
        if not isinstance(text, str) or not text.strip():
            return text

        # 1) 机构/实体名替换（基于 bank.txt）
        if self.entity_words:
            entities = list(self.entity_words)

            def _entity_repl(match):
                if self._rand(rng) < self.prob:
                    return self._choice(entities, rng)
                return match.group()

            # 仅替换在词库中出现的实体
            for ent in entities:
                pattern = re.compile(re.escape(ent))
                text = pattern.sub(_entity_repl, text)

        # 2) 日期替换（先于纯数字替换，避免破坏"12月26日"）
        date_pat = re.compile(r"(\d{1,2})月(\d{1,2})日")

        def _date_repl(match):
            if self._rand(rng) < self.prob:
                month = self._randint(1, 12, rng)
                day = self._randint(1, 28, rng)
                return f"{month}月{day}日"
            return match.group()

        text = date_pat.sub(_date_repl, text)

        # 3) 纯数字替换：保持长度一致，避免破坏卡号/金额
        digit_pat = re.compile(r"\d+")

        def _digit_repl(match):
            if self._rand(rng) < self.prob:
                s = match.group()
                L = len(s)
                if L == 1:
                    return str(self._randint(0, 9, rng))
                if L == 2:
                    return str(self._randint(0, 99, rng)).zfill(2)
                # 长数字（卡号、电话）只修改 1-2 位
                chars = list(s)
                change_count = min(L, self._randint(1, 2, rng))
                positions = self._sample(range(L), change_count, rng)
                for p in positions:
                    chars[p] = str(self._randint(0, 9, rng))
                return ''.join(chars)
            return match.group()

        text = digit_pat.sub(_digit_repl, text)
        return text
