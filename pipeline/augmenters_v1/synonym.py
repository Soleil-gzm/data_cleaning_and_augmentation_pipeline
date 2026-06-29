"""
同义词替换增强器
"""
from typing import Optional
import random
from .base import BaseAugmenter
from .utils import tokenize, load_synonym_dict


class SynonymAugmenter(BaseAugmenter):
    def _load_resources(self):
        dict_path = self.config.get("dict_path", "resources/synonyms.txt")
        self.synonym_dict = load_synonym_dict(dict_path)
        self.prob = self.config.get("prob", 0.3)

    def apply(self, text: str, rng: Optional[random.Random] = None) -> str:
        self.initialize()
        if not self.synonym_dict:
            return text
        tokens = tokenize(text)
        if not tokens:
            return text
        new_tokens = []
        for token in tokens:
            if token in self.synonym_dict and self._rand(rng) < self.prob:
                synonyms = self.synonym_dict[token]
                if synonyms:
                    new_tokens.append(self._choice(synonyms, rng))
                    continue
            new_tokens.append(token)
        return ''.join(new_tokens)
