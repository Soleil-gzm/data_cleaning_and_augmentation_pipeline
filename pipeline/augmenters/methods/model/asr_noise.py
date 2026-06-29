"""
ASR 噪声增强器（model）
======================
- 需要加载 SentenceTransformer 语义编码器 + 拼音/异常词 pickle
- 多位置并行（最多 max_operations 个互不相邻位置）
- 替换/插入两种操作（insert_prob 控制）
- 极性保护（避免肯定/否定词被翻转）
- 重试机制（retry_times，若结果未变则重采样）

资源参数（config 支持）：
    vectors_path   / pinyin_path / prev_map_path / model_path / model_name
    prob / alpha / max_operations / insert_prob / retry_times
    encoder        : 可注入已预加载的 SentenceTransformer（用于多进程共享）
"""
import pickle
import re
import random
import numpy as np
from pathlib import Path
from typing import Optional, List

from ...base import BaseAugmenter


AFFIRMATIVE_WORDS = {
    "是", "有", "能", "可以", "行", "好", "对", "是的", "没错",
    "肯定", "必须", "需要", "会", "应该",
}

NEGATIVE_WORDS = {
    "不", "没", "无", "别", "不要", "不用", "不行", "不是", "没有",
    "不能", "不可以", "否定", "不会", "不该",
}


class AsrNoiseAugmenter(BaseAugmenter):
    def __init__(self, config):
        super().__init__(config)
        self._encoder_injected = config.get("encoder", None)

    def _resolve(self, p):
        if p is None:
            return None
        path = Path(p)
        if not path.is_absolute():
            root = Path(__file__).resolve().parents[4]  # pipeline/augmenters/methods/model -> project root
            path = root / path
        return str(path)

    def _load_resources(self):
        # 若有主进程注入的共享资源，优先使用（避免重复加载）
        shared = self.config.get("shared_resources")
        if shared and all(
            shared.get(k) for k in (
                "asr_noise.abnormal_words",
                "asr_noise.abnormal_vectors",
                "asr_noise.pinyin_dict",
            )
        ):
            self.abnormal_words = shared["asr_noise.abnormal_words"]
            self.abnormal_vectors = shared["asr_noise.abnormal_vectors"]
            self.word_to_idx = shared.get(
                "asr_noise.word_to_idx",
                {w: i for i, w in enumerate(self.abnormal_words)},
            )
            self.pinyin_dict = shared["asr_noise.pinyin_dict"]
            self.prev_to_abnormals = shared.get("asr_noise.prev_to_abnormals", {})
            saved_cfg = shared.get("asr_noise.config", {})
            self.prob = float(saved_cfg.get("prob", self.config.get("prob", 0.5)))
            self.alpha = float(saved_cfg.get("alpha", self.config.get("alpha", 0.7)))
            self.max_operations = int(saved_cfg.get("max_operations", self.config.get("max_operations", 2)))
            self.insert_prob = float(saved_cfg.get("insert_prob", self.config.get("insert_prob", 0.2)))
            self.retry_times = int(saved_cfg.get("retry_times", self.config.get("retry_times", 3)))
            self.dim = int(saved_cfg.get("dim", self.abnormal_vectors.shape[1]))

            # 共享资源仅包含权重/词典，encoder 仍需在子进程内懒加载（可选开启）
            if self.config.get("load_encoder_with_shared", False):
                try:
                    self._load_encoder(
                        self._resolve(self.config.get("model_path")),
                        self.config.get("model_name", "paraphrase-multilingual-MiniLM-L12-v2"),
                    )
                except Exception as e:
                    self._ready = False
                    self._load_error = f"shared encoder 加载失败: {e}"
                    return
            self._ready = True
            self._load_error = None
            return

        # 常规磁盘加载路径
        vectors_path = self._resolve(self.config.get("vectors_path"))
        pinyin_path = self._resolve(self.config.get("pinyin_path"))
        prev_map_path = self._resolve(self.config.get("prev_map_path"))
        model_path = self._resolve(self.config.get("model_path"))
        model_name = self.config.get("model_name", "paraphrase-multilingual-MiniLM-L12-v2")

        if vectors_path is None or pinyin_path is None:
            self._ready = False
            self._load_error = "asr_noise 缺少 vectors_path / pinyin_path"
            return

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

        try:
            self._load_encoder(model_path, model_name)
        except Exception as e:
            self._ready = False
            self._load_error = f"asr_noise encoder 加载失败: {e}"
            return

        self.dim = self.abnormal_vectors.shape[1]
        self.prob = float(self.config.get("prob", 0.5))
        self.alpha = float(self.config.get("alpha", 0.7))
        self.max_operations = int(self.config.get("max_operations", 2))
        self.insert_prob = float(self.config.get("insert_prob", 0.2))
        self.retry_times = int(self.config.get("retry_times", 3))
        self._ready = True
        self._load_error = None

    def _load_encoder(self, model_path: Optional[str], model_name: str):
        if self._encoder_injected is not None:
            self.encoder = self._encoder_injected
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "asr_noise 需要 sentence_transformers 库；请先 `pip install sentence-transformers`，"
                "或在配置中把 asr_noise.enabled 设为 false"
            ) from e
        if model_path and Path(model_path).exists():
            self.encoder = SentenceTransformer(model_path)
        else:
            self.encoder = SentenceTransformer(model_name)

    # ---------- 工具 ----------
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
        if not getattr(self, "_ready", False):
            return []
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
        if not getattr(self, "_ready", False):
            return sentence

        words = list(jieba.lcut(sentence))
        if len(words) < 2:
            return sentence

        candidate_indices = []
        for i in range(1, len(words)):
            if words[i - 1] in self.prev_to_abnormals:
                candidate_indices.append(i)
        if not candidate_indices:
            return sentence

        max_ops = min(self.max_operations, len(candidate_indices))
        selected = []
        shuffled = self._sample(candidate_indices, len(candidate_indices), rng)
        for idx in shuffled:
            if not selected or all(abs(idx - x) >= 2 for x in selected):
                selected.append(idx)
                if len(selected) >= max_ops:
                    break

        operations = []
        for pos in selected:
            prev_word = words[pos - 1]
            target_word = words[pos]

            if self._rand(rng) > self.prob:
                continue

            candidates = self.find_best_abnormals(target_word, prev_word=prev_word, top_k=5)
            if not candidates:
                continue

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
        if not getattr(self, "_ready", False):
            return text

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
