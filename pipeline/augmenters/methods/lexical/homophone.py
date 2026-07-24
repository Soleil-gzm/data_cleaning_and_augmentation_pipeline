"""
同音字替换增强器（lexical）
"""
from typing import Optional

from ...base import BaseAugmenter
from ...utils import tokenize, load_homophone_dict
from ....utils.random_utils import rand, choice


class HomophoneAugmenter(BaseAugmenter):
    def _load_resources(self):
        dict_path = self.config.get("dict_path", "resources/Homophone_tab.txt")
        self.homophone_dict = load_homophone_dict(dict_path)
        self.prob = float(self.config.get("prob", 0.2))

    def apply(self, text: str, rng=None) -> str:
        self.initialize()
        if not isinstance(text, str) or not text.strip():
            return text
        if not self.homophone_dict:
            return text
        tokens = tokenize(text)
        if not tokens:
            return text
        new_tokens = []
        for token in tokens:
            if token in self.homophone_dict and rand(rng=rng) < self.prob:
                homophones = self.homophone_dict[token]
                if homophones:
                    new_tokens.append(choice(homophones, rng=rng))
                    continue
            new_tokens.append(token)
        return ''.join(new_tokens)
