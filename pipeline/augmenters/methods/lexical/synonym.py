"""
同义词替换增强器（lexical）
"""

from pathlib import Path
from typing import Optional

from ...base import BaseAugmenter
from ...utils import tokenize, load_synonym_dict
from ....utils.random_utils import rand, choice


class SynonymAugmenter(BaseAugmenter):
    def _resolve(self, p):
        if p is None:
            return None
        path = Path(p)
        if not path.is_absolute():
            root = Path(__file__).resolve().parents[4]
            path = root / path
        return str(path)

    def _load_resources(self):
        dict_path = self._resolve(
            self.config.get("dict_path", "resources/synonyms.txt")
        )
        self.synonym_dict = load_synonym_dict(dict_path)
        self.prob = float(self.config.get("prob", 0.3))

    def apply(self, text: str, rng=None) -> str:
        self.initialize()
        if not isinstance(text, str) or not text.strip():
            return text
        if not self.synonym_dict:
            return text
        tokens = tokenize(text)
        if not tokens:
            return text
        new_tokens = []
        for token in tokens:
            if token in self.synonym_dict and rand(rng=rng) < self.prob:
                synonyms = self.synonym_dict[token]
                if synonyms:
                    new_tokens.append(choice(synonyms, rng=rng))
                    continue
            new_tokens.append(token)
        return "".join(new_tokens)
