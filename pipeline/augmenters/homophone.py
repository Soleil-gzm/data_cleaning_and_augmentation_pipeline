"""
同音字替换（基于预定义同音词库）
"""
import random
from .base import BaseAugmenter
from .utils import tokenize, load_homophone_dict


class HomophoneAugmenter(BaseAugmenter):
    def _load_resources(self):
        dict_path = self.config.get("dict_path", "resources/Homophone_tab.txt")
        self.homophone_dict = load_homophone_dict(dict_path)
        self.prob = self.config.get("prob", 0.2)

    def apply(self, text: str) -> str:
        self.initialize()
        if not self.homophone_dict:
            return text
        tokens = tokenize(text)
        if not tokens:
            return text
        new_tokens = []
        for token in tokens:
            if token in self.homophone_dict and random.random() < self.prob:
                homophones = self.homophone_dict[token]
                if homophones:
                    new_tokens.append(random.choice(homophones))
                    continue
            new_tokens.append(token)
        return ''.join(new_tokens)