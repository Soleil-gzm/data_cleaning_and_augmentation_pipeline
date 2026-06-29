"""
插入语气词增强器
- 支持句首插入或句中插入（按标点位置，若偏后半则改为句首）
- 使用独立 FILLERS / TAILS 词库
"""
from typing import Optional
import re
import random
from .base import BaseAugmenter


class InsertFillerAugmenter(BaseAugmenter):
    FILLERS = ["嗯", "那个", "就是", "呃", "啊", "这个", "其实"]
    TAILS = ["吧", "啊", "哦", "呗", "啦", "呀", "嘛", "呐", "哈", "了", "吗", "呢"]

    def __init__(self, config):
        super().__init__(config)
        self.prob = config.get("prob", 0.3)

    def apply(self, text: str, rng: Optional[random.Random] = None) -> str:
        if not isinstance(text, str) or len(text.strip()) == 0:
            return text

        filler = self._choice(self.FILLERS, rng)

        # 60% 概率直接句首插入
        if self._rand(rng) < 0.6:
            return f"{filler}，{text}"

        # 否则尝试句中按标点插入
        match = re.search(r"[，,。？!]", text)
        if match:
            pos = match.end()
            if pos > len(text) * 0.6:
                return f"{filler}，{text}"
            return text[:pos] + filler + "，" + text[pos:]

        # 没有标点时，退化为句首
        return f"{filler}，{text}"
