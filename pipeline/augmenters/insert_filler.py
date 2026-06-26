"""
插入语气词（在句子开头或结尾插入 "嗯"、"啊" 等）
"""
import random
from .base import BaseAugmenter
from .utils import split_sentences


class InsertFillerAugmenter(BaseAugmenter):
    FILLERS = ["嗯", "啊", "呃", "那个", "就是说", "这个", "其实"]

    def __init__(self, config):
        super().__init__(config)
        self.prob = config.get("prob", 0.3)

    def apply(self, text: str) -> str:
        sentences = split_sentences(text)
        if not sentences:
            return text
        new_sentences = []
        for sent in sentences:
            if random.random() < self.prob:
                filler = random.choice(self.FILLERS)
                if random.random() < 0.5:
                    sent = filler + "，" + sent
                else:
                    sent = sent + "，" + filler
            new_sentences.append(sent)
        return ''.join(new_sentences)