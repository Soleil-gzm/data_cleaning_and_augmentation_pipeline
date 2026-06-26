"""
ASR 噪声增强器（延迟加载模型）
完全从 common/asr_noise_augmenter.py 迁移
"""
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from .base import BaseAugmenter


class AsrNoiseAugmenter(BaseAugmenter):
    """
    ASR 噪声增强器，基于前置词和语义+拼音混合匹配。
    延迟加载模型，仅在首次调用时初始化。
    """

    def _load_resources(self):
        """加载 ASR 模型和字典"""
        vectors_path = self.config.get("vectors_path")
        pinyin_path = self.config.get("pinyin_path")
        prev_map_path = self.config.get("prev_map_path")
        model_path = self.config.get("model_path")
        model_name = self.config.get("model_name", "paraphrase-multilingual-MiniLM-L12-v2")

        # 解析路径
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

        # 加载向量
        with open(vectors_path, 'rb') as f:
            vec_data = pickle.load(f)
        self.abnormal_words = vec_data['words']
        self.abnormal_vectors = vec_data['vectors']
        self.word_to_idx = {w: i for i, w in enumerate(self.abnormal_words)}

        # 加载拼音
        with open(pinyin_path, 'rb') as f:
            self.pinyin_dict = pickle.load(f)

        # 加载前置词映射
        if prev_map_path and Path(prev_map_path).exists():
            with open(prev_map_path, 'rb') as f:
                self.prev_to_abnormals = pickle.load(f)
        else:
            self.prev_to_abnormals = {}

        # 加载编码器
        self._load_encoder(model_path, model_name)

        self.dim = self.abnormal_vectors.shape[1]
        self.prob = self.config.get("prob", 0.5)
        self.alpha = self.config.get("alpha", 0.7)

    def _load_encoder(self, model_path: Optional[str], model_name: str):
        """加载 SentenceTransformer 编码器"""
        try:
            from sentence_transformers import SentenceTransformer
            if model_path and Path(model_path).exists():
                self.encoder = SentenceTransformer(model_path)
            else:
                self.encoder = SentenceTransformer(model_name)
        except ImportError as e:
            raise ImportError(
                "ASR 增强需要 sentence_transformers 库，请安装: pip install sentence_transformers"
            ) from e

    def _pinyin_similarity(self, w1: str, w2: str) -> float:
        """计算两个词的拼音相似度（基于编辑距离）"""
        p1 = self.pinyin_dict.get(w1, '')
        p2 = self.pinyin_dict.get(w2, '')
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
            # 降级：简单字符匹配
            common = sum(1 for a, b in zip(p1, p2) if a == b)
            return common / max_len

    def _cosine_sim(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """计算余弦相似度"""
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
        """
        查找最佳异常词替换候选
        """
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

    def apply(self, text: str) -> str:
        """
        对文本应用 ASR 噪声增强
        """
        self.initialize()
        if not text.strip():
            return text

        # 分词
        try:
            import jieba
            tokens = list(jieba.cut(text))
        except ImportError:
            return text

        if len(tokens) < 2:
            return text

        # 遍历 token，查找可替换的词
        new_tokens = []
        for i, token in enumerate(tokens):
            # 检查是否应替换
            if random.random() > self.prob:
                new_tokens.append(token)
                continue

            prev_word = tokens[i-1] if i > 0 else None
            # 查找候选异常词
            candidates = self.find_best_abnormals(token, prev_word, top_k=3)
            if candidates:
                new_tokens.append(random.choice(candidates))
            else:
                new_tokens.append(token)

        return ''.join(new_tokens)