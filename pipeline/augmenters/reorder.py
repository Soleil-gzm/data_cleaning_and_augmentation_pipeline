"""
语序打乱增强器
- 否定词保护：跳过包含否定词的句子，避免语义反转
- 模式 1：交换逗号前后分句
- 模式 2：谓语前置（匹配 "我/你 + 副词 + 动词 + 了/过" 结构）
"""
from typing import Optional
import re
import random
from .base import BaseAugmenter


NEGATION_WORDS = {"不", "没", "无", "别", "不要", "不用", "未曾"}


class ReorderAugmenter(BaseAugmenter):
    def apply(self, text: str, rng: Optional[random.Random] = None) -> str:
        if not isinstance(text, str) or len(text.strip()) < 5:
            return text
        if any(neg in text for neg in NEGATION_WORDS):
            return text

        end_punct = ""
        if text and text[-1] in "。！？!?":
            end_punct = text[-1]
            text = text[:-1].rstrip()

        # 模式 1：交换逗号前后
        if "，" in text:
            parts = text.split("，", 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                new_sent = f"{parts[1].strip()}，{parts[0].strip()}"
                return new_sent + end_punct
        if "," in text:
            parts = text.split(",", 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                new_sent = f"{parts[1].strip()}，{parts[0].strip()}"
                return new_sent + end_punct

        # 模式 2：谓语前置
        match = re.match(r"^(我|你)(已经|也|就|都)?(\w+?)(了|过)?(.*)$", text)
        if match:
            subject = match.group(1)
            adverb = match.group(2) or ""
            verb = match.group(3)
            aspect = match.group(4) or ""
            rest = match.group(5).strip()
            if verb and len(verb) >= 1:
                new_sent = f"{verb}{aspect}{rest}，{subject}{adverb}"
                new_sent = re.sub(r"\s+", "", new_sent)
                return new_sent + end_punct

        return text + end_punct
