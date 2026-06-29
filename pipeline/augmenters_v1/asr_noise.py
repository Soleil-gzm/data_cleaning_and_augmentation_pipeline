"""
ASR 噪声增强器（恢复完整核心算法）
- 多位置并行（最多 MAX_OPERATIONS 个互不相邻位置）
- 替换/插入两种操作（INSERT_PROB 控制）
- 极性保护（避免肯定/否定词被翻转）
- 重试机制（RETRY_TIMES，若结果未变则重采样）
"""
import pickle
import re
import random
import numpy as np
from pathlib import Path
from typing import Optional, List

from .base import BaseAugmenter


AFFIRMATIVE_WORDS = {
    "是", "有", "能", "可以", "行", "好", "对", "是的", "没错",
    "肯定", "必须", "需要", "会", "应该",
}

NEGATIVE_WORDS = {
    "不", "没", "无", "别", "不要", "不用", "不行", "不是", "没有",
    "不能", "不可以", "否定", "不会", "不该",
}


class AsrNoiseAugmenter(BaseAugmenter):
    def _load_resources(self):
        vectors_path = self.config.get("vectors_path")
        pinyin_path = self.config.get("pinyin_path")
        prev_map_path = self.config.get("prev_map_path")
        model_path = self.config.get("model_path")
        model_name = self.config.get("model_name", "paraphrase-multilingual-MiniLM-L12-v2")

        project_root = Path(__file__).parent.parent.parent

        def resolve(p):
            if p is None:
                return None
            path = Path(p)
            if not path.is_absolute():
                path = project_root / path
            return str(path)

        vectors_path = resolve(vectors_path)
        pinyin_path = resolve(pinyin_path)
        prev_map_path = resolve(prev_map_path)
        model_path = resolve(model_path)

        with open(vectors_path, 'rb') as f:
            vec_data = pickle.load(f)
        self.abnormal_words = vec_data['words']
        self.abnormal_vectors = vec_data['vectors']
        self.word_to_idx = {w: i for i, w in enumerate(self.abnormal_words)}

        with open(pinyin_path, 'rb') as f:
            self.pinyin_dict = pickle.load(f)

        if prev_map_path and Path(prev_map_path).exists():
            with open(prev_map_path, 'rb') as f:
                self.prev_to_abnormals = pickle.load(f)
        else:
            self.prev_to_abnormals = {}

        self._load_encoder(model_path, model_name)

        self.dim = self.abnormal_vectors.shape[1]
        self.prob = float(self.config.get("prob", 0.5))
        self.alpha = float(self.config.get("alpha", 0.7))
        self.max_operations = int(self.config.get("max_operations", 2))
        self.insert_prob = float(self.config.get("insert_prob", 0.2))
        self.retry_times = int(self.config.get("retry_times", 3))

    def _load_encoder(self, model_path: Optional[str], model_name: str):
        from sentence_transformers import SentenceTransformer
        if model_path and Path(model_path).exists():
            self.encoder = SentenceTransformer(model_path)
        else:
            self.encoder = SentenceTransformer(model_name)

    def _pinyin_similarity(self, w1: str, w2: str) -> float:
        p1 = self.pinyin_dict.get(w1, '')
        p2 = self.pinyin_dict.get(w2, '')
        if not p1 or not p2:
            return 0.0
        max_len = max(len(p1), len(p2))
        if max_len == 0:
            return 1.0
        try:
            import Levenshtein  # type: ignore
            dist = Levenshtein.distance(p1, p2)
            return 1 - dist / max_len
        except ImportError:
            common = sum(1 for a, b in zip(p1, p2) if a == b)
            return common / max_len

    def _cosine_sim(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def find_best_abnormals(
        self,
        target_word: str,
        prev_word: Optional[str] = None,
        top_k: int = 5,
    ) -> List[str]:
        self.initialize()
        if prev_word and prev_word in self.prev_to_abnormals:
            candidates = self.prev_to_abnormals[prev_word]
        else:
            candidates = self.abnormal_words
        if not candidates:
            return []

        target_vec = self.encoder.encode([target_word])[0]
        scores = []
        for ab in candidates:
            idx = self.word_to_idx.get(ab)
            if idx is None:
                continue
            sem_sim = self._cosine_sim(target_vec, self.abnormal_vectors[idx])
            pin_sim = self._pinyin_similarity(target_word, ab)
            combined = self.alpha * pin_sim + (1 - self.alpha) * sem_sim
            scores.append((ab, combined))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [ab for ab, _ in scores[:top_k]]

    # ---------- 核心增强算法 ----------
    def _enhance_once(self, sentence: str, rng=None) -> str:
        try:
            import jieba
        except ImportError:
            return sentence

        words = list(jieba.lcut(sentence))
        if len(words) < 2:
            return sentence

        # 1) 找所有可操作位置（必须满足前置词在 prev_to_abnormals 中）
        candidate_indices = []
        for i in range(1, len(words)):
            if words[i - 1] in self.prev_to_abnormals:
                candidate_indices.append(i)
        if not candidate_indices:
            return sentence

        # 2) 随机选最多 max_operations 个互不相邻的位置
        max_ops = min(self.max_operations, len(candidate_indices))
        selected = []
        shuffled = self._sample(candidate_indices, len(candidate_indices), rng)
        for idx in shuffled:
            if not selected or all(abs(idx - x) >= 2 for x in selected):
                selected.append(idx)
                if len(selected) >= max_ops:
                    break

        # 3) 为每个选中位置决定替换/插入，并做极性保护
        operations = []
        for pos in selected:
            prev_word = words[pos - 1]
            target_word = words[pos]

            # 按概率决定本次是否操作
            if self._rand(rng) > self.prob:
                continue

            candidates = self.find_best_abnormals(target_word, prev_word=prev_word, top_k=5)
            if not candidates:
                continue

            # 极性检测
            target_polarity = None
            if target_word in AFFIRMATIVE_WORDS:
                target_polarity = "affirmative"
            elif target_word in NEGATIVE_WORDS:
                target_polarity = "negative"

            chosen = None
            for _ in range(5):
                cand = self._choice(candidates, rng)
                if target_polarity is None:
                    chosen = cand
                    break
                if cand in AFFIRMATIVE_WORDS:
                    cand_polarity = "affirmative"
                elif cand in NEGATIVE_WORDS:
                    cand_polarity = "negative"
                else:
                    cand_polarity = None
                if cand_polarity == target_polarity or cand_polarity is None:
                    chosen = cand
                    break
            if chosen is None:
                continue

            is_insert = self._rand(rng) < self.insert_prob
            operations.append((pos, chosen, is_insert))

        if not operations:
            return sentence

        # 4) 从后往前应用操作，避免索引偏移
        new_words = words[:]
        for pos, new_word, is_insert in sorted(operations, key=lambda x: x[0], reverse=True):
            if is_insert:
                new_words.insert(pos, new_word)
            else:
                new_words[pos] = new_word
        return ''.join(new_words)

    def apply(self, text: str, rng: Optional[random.Random] = None) -> str:
        self.initialize()
        if not isinstance(text, str) or not text.strip():
            return text

        # 若文本中包含多句（按 / 分割），逐句增强后合并
        if '/' in text or '／' in text:
            parts = re.split(r'[／/]', text)
            enhanced = [self._apply_single(p, rng) for p in parts]
            return '/'.join(enhanced)

        return self._apply_single(text, rng)

    def _apply_single(self, text: str, rng) -> str:
        original = text
        for _ in range(self.retry_times):
            result = self._enhance_once(original, rng=rng)
            if result != original:
                return result
        return original
