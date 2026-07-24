"""
插入语气词增强器（lexical）
- 支持句首插入或句中插入（按标点位置，若偏后半则改为句首）
- 使用独立 FILLERS / TAILS 词库
"""
from typing import Optional
import re

from ...base import BaseAugmenter
from ....utils.random_utils import rand, choice


class InsertFillerAugmenter(BaseAugmenter):
    FILLERS = ["嗯", "那个", "就是", "呃", "啊", "这个", "其实"]
    TAILS = ["吧", "啊", "哦", "呗", "啦", "呀", "嘛", "呐", "哈", "了", "吗", "呢"]

    def __init__(self, config):
        super().__init__(config)
        self.prob = float(config.get("prob", 0.3))
        self.fillers = list(config.get("fillers", self.FILLERS))

    def apply(self, text: str, rng=None) -> str:
        if not isinstance(text, str) or len(text.strip()) == 0:
            return text

        filler = choice(self.fillers, rng=rng)

        if rand(rng=rng) < 0.6:
            return f"{filler}，{text}"

        match = re.search(r"[，,。？!]", text)
        if match:
            pos = match.end()
            if pos > len(text) * 0.6:
                return f"{filler}，{text}"
            return text[:pos] + filler + "，" + text[pos:]

        return f"{filler}，{text}"
