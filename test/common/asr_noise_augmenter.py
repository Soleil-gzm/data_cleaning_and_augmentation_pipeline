# common/asr_noise_augmenter.py
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pypinyin import pinyin, Style
import Levenshtein

class AsrNoiseAugmenter:
    def __init__(self, vectors_path, pinyin_path, prev_map_path=None,
                 model_path=None, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """
        :param vectors_path: abnormal_vectors.pkl 文件路径
        :param pinyin_path: abnormal_pinyin.pkl 文件路径
        :param prev_map_path: prev_to_abnormals.pkl 文件路径（可选）
        :param model_path: 本地 SentenceTransformer 模型文件夹路径（推荐）
        :param model_name: 模型名称（当 model_path 为 None 时使用，可能触发联网）
        """
        vectors_path = Path(vectors_path)
        pinyin_path = Path(pinyin_path)
        
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
        
        # 加载编码器：优先使用本地路径
        if model_path and Path(model_path).exists():
            self.encoder = SentenceTransformer(str(model_path))
        else:
            # 如果 model_path 无效，回退到 model_name（但会尝试联网，不推荐）
            self.encoder = SentenceTransformer(model_name)
        
        self.dim = self.abnormal_vectors.shape[1]
    
    # 以下方法保持不变
    def _pinyin_similarity(self, w1, w2):
        p1 = self.pinyin_dict.get(w1, '')
        p2 = self.pinyin_dict.get(w2, '')
        if not p1 or not p2:
            return 0.0
        max_len = max(len(p1), len(p2))
        if max_len == 0:
            return 1.0
        dist = Levenshtein.distance(p1, p2)
        return 1 - dist / max_len
    
    def _cosine_sim(self, vec_a, vec_b):
        return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    
    def find_best_abnormals(self, target_word, prev_word=None, top_k=5, alpha=0.7):
        if prev_word and prev_word in self.prev_to_abnormals:
            candidates = self.prev_to_abnormals[prev_word]
        else:
            candidates = self.abnormal_words
        if not candidates:
            return []
        target_vec = self.encoder.encode([target_word])[0]
        scores = []
        for ab in candidates:
            idx = self.word_to_idx[ab]
            sem_sim = self._cosine_sim(target_vec, self.abnormal_vectors[idx])
            pin_sim = self._pinyin_similarity(target_word, ab)
            combined = alpha * pin_sim + (1 - alpha) * sem_sim
            scores.append((ab, combined))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [ab for ab, _ in scores[:top_k]]