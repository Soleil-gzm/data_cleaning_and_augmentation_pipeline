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
    device         : 指定设备（如 "cuda:7"、"cpu"），默认自动检测
    encoder        : 可注入已预加载的 SentenceTransformer（用于多进程共享）
"""

import pickle
import re
import random
import numpy as np
from pathlib import Path
from typing import Optional, List

from ...base import BaseAugmenter

# ── 模块级缓存（Worker 进程内共享，避免每个实例重复加载） ──
_module_cache = {
    "word_to_vec": None,   # 主进程预编码的 {word: np.ndarray}
    "encoder": None,       # SentenceTransformer 实例（仅在无预编码时使用）
}

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
            root = Path(__file__).resolve().parents[4]
            path = root / path
        return str(path)

    @staticmethod
    def _has_shared(shared, key):
        """安全检查共享资源中某个 key 是否存在且非空"""
        val = shared.get(key)
        if val is None:
            return False
        if isinstance(val, np.ndarray):
            return val.size > 0
        if isinstance(val, (list, tuple, dict, set)):
            return len(val) > 0
        return bool(val)

    def _load_resources(self):
        shared = self.config.get("shared_resources")

        # ── 路径 A：从共享资源加载（Worker 进程）──
        if shared and all(
            self._has_shared(shared, k)
            for k in (
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
            self.prob = float(saved_cfg.get("prob", self.config.get("prob", 1.0)))
            self.alpha = float(saved_cfg.get("alpha", self.config.get("alpha", 0.7)))
            self.max_operations = int(
                saved_cfg.get("max_operations", self.config.get("max_operations", 2))
            )
            self.insert_prob = float(
                saved_cfg.get("insert_prob", self.config.get("insert_prob", 0.2))
            )
            self.retry_times = int(
                saved_cfg.get("retry_times", self.config.get("retry_times", 3))
            )
            self.dim = int(saved_cfg.get("dim", self.abnormal_vectors.shape[1]))

            # 从模块级缓存或共享资源获取预编码向量
            self._setup_word_to_vec()

            # 如果没有预编码向量，需要加载 encoder 作为 fallback
            if self._word_to_vec is None:
                try:
                    model_path = self._resolve(self.config.get("model_path"))
                    model_name = self.config.get(
                        "model_name", "paraphrase-multilingual-MiniLM-L12-v2"
                    )
                    self._load_encoder(model_path, model_name)
                except Exception as e:
                    self._ready = False
                    self._load_error = f"asr_noise encoder 加载失败: {e}"
                    return

            self._ready = True
            self._load_error = None
            return

        # ── 路径 B：从磁盘加载（主进程 / 串行模式）──
        vectors_path = self._resolve(self.config.get("vectors_path"))
        pinyin_path = self._resolve(self.config.get("pinyin_path"))
        prev_map_path = self._resolve(self.config.get("prev_map_path"))
        model_path = self._resolve(self.config.get("model_path"))
        model_name = self.config.get(
            "model_name", "paraphrase-multilingual-MiniLM-L12-v2"
        )

        if vectors_path is None or pinyin_path is None:
            self._ready = False
            self._load_error = "asr_noise 缺少 vectors_path / pinyin_path"
            return

        with open(vectors_path, "rb") as f:
            vec_data = pickle.load(f)
        self.abnormal_words = vec_data["words"]
        self.abnormal_vectors = vec_data["vectors"]
        self.word_to_idx = {w: i for i, w in enumerate(self.abnormal_words)}

        with open(pinyin_path, "rb") as f:
            self.pinyin_dict = pickle.load(f)

        if prev_map_path and Path(prev_map_path).exists():
            with open(prev_map_path, "rb") as f:
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
        self.prob = float(self.config.get("prob", 1.0))
        self.alpha = float(self.config.get("alpha", 0.7))
        self.max_operations = int(self.config.get("max_operations", 2))
        self.insert_prob = float(self.config.get("insert_prob", 0.2))
        self.retry_times = int(self.config.get("retry_times", 3))
        self._ready = True
        self._load_error = None

    def _setup_word_to_vec(self):
        """
        设置 word_to_vec：优先从模块级缓存读取，其次从 config 注入读取。
        Worker 进程中，主进程预编码的向量通过 task dict 注入到 config。
        """
        # 1. 检查模块级缓存（同一进程内的多个实例共享）
        if _module_cache["word_to_vec"] is not None:
            self._word_to_vec = _module_cache["word_to_vec"]
            return

        # 2. 检查 config 注入（Worker 进程首次调用时）
        injected = self.config.get("asr_noise.word_to_vec")
        if injected:
            self._word_to_vec = injected
            _module_cache["word_to_vec"] = injected
            return

        # 3. 无预编码向量
        self._word_to_vec = None

    def _load_encoder(self, model_path: Optional[str], model_name: str):
        """加载 SentenceTransformer 模型（仅在无预编码向量时调用）"""
        if self._encoder_injected is not None:
            self.encoder = self._encoder_injected
            return

        # 检查模块级缓存
        if _module_cache["encoder"] is not None:
            self.encoder = _module_cache["encoder"]
            return

        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError as e:
            raise RuntimeError(
                "asr_noise 需要 sentence_transformers 库；请先 `pip install sentence-transformers`，"
                "或在配置中把 asr_noise.enabled 设为 false"
            ) from e

        # 自动检测 GPU
        device = self.config.get("device", None)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_path and Path(model_path).exists():
            self.encoder = SentenceTransformer(model_path, device=device)
        else:
            self.encoder = SentenceTransformer(model_name, device=device)

        # 缓存到模块级
        _module_cache["encoder"] = self.encoder

    def _get_target_vec(self, target_word: str) -> Optional[np.ndarray]:
        """获取 target_word 的语义向量：优先查表，其次用 encoder 编码"""
        # 1. 查预编码表
        if self._word_to_vec is not None and target_word in self._word_to_vec:
            return self._word_to_vec[target_word]

        # 2. 用 encoder 编码（仅在无预编码表时）
        if self.encoder is not None:
            return self.encoder.encode([target_word])[0]

        # 3. 无 encoder 也无预编码，无法计算
        return None

    def _pinyin_similarity(self, w1: str, w2: str) -> float:
        p1 = self.pinyin_dict.get(w1, "")
        p2 = self.pinyin_dict.get(w2, "")
        if not p1 or not p2:
            return 0.0
        max_len = max(len(p1), len(p2))
        if max_len == 0:
            return 1.0
        try:
            import Levenshtein
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

        target_vec = self._get_target_vec(target_word)
        if target_vec is None:
            return []

        # 先尝试上下文相关候选（prev_to_abnormals）
        candidates_pool = None
        if prev_word and prev_word in self.prev_to_abnormals:
            candidates_pool = self.prev_to_abnormals[prev_word]

        def _score(candidates):
            if not candidates:
                return []
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
            return scores

        scores = _score(candidates_pool)
        if len(scores) < top_k:
            fallback = _score(self.abnormal_words)
            existing = {w for w, _ in scores}
            for w, s in fallback:
                if w not in existing:
                    scores.append((w, s))
            scores.sort(key=lambda x: x[1], reverse=True)

        return [ab for ab, _ in scores[:top_k]]

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

        # 第一优先级：有上下文的候选
        context_candidates = []
        for i in range(1, len(words)):
            if words[i - 1] in self.prev_to_abnormals:
                context_candidates.append(i)

        # 第二优先级：全量词表回退
        if not context_candidates:
            context_candidates = list(range(len(words)))

        if not context_candidates:
            return sentence

        max_ops = min(self.max_operations, len(context_candidates))
        selected = []
        shuffled = self._sample(context_candidates, len(context_candidates), rng)
        for idx in shuffled:
            if not selected or all(abs(idx - x) >= 2 for x in selected):
                selected.append(idx)
                if len(selected) >= max_ops:
                    break

        operations = []
        for pos in selected:
            target_word = words[pos]
            prev_word = words[pos - 1] if pos > 0 else None

            if self._rand(rng) > self.prob:
                continue

            candidates = self.find_best_abnormals(
                target_word, prev_word=prev_word, top_k=5
            )
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
        for pos, new_word, is_insert in sorted(
            operations, key=lambda x: x[0], reverse=True
        ):
            if is_insert:
                new_words.insert(pos, new_word)
            else:
                new_words[pos] = new_word
        return "".join(new_words)

    def apply(self, text: str, rng: Optional[random.Random] = None) -> str:
        self.initialize()
        if not isinstance(text, str) or not text.strip():
            return text
        if not getattr(self, "_ready", False):
            return text

        if "/" in text or "／" in text:
            parts = re.split(r"[／/]", text)
            enhanced = [self._apply_single(p, rng) for p in parts]
            return "/".join(enhanced)
        return self._apply_single(text, rng)

    def _apply_single(self, text: str, rng) -> str:
        original = text
        for i in range(self.retry_times):
            result = self._enhance_once(original, rng=rng)
            if result != original:
                return result
        return original
