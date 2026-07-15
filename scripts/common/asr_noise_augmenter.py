# common/asr_noise_augmenter.py
import pickle
import numpy as np
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pypinyin import pinyin, Style
import Levenshtein

_asr_global_stats = {
    "encode_calls": 0,
    "encode_total_time": 0.0,
    "find_best_calls": 0,
    "find_best_total_time": 0.0,
    "similarity_total_time": 0.0,
}

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
            self.encoder = SentenceTransformer(model_name)
        
        self.dim = self.abnormal_vectors.shape[1]
        
        # 向量缓存：避免对同一个词重复 encode
        self._encode_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
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
        _start = time.time()
        
        if prev_word and prev_word in self.prev_to_abnormals:
            candidates = self.prev_to_abnormals[prev_word]
        else:
            candidates = self.abnormal_words
        if not candidates:
            return []
        
        _encode_start = time.time()
        if target_word in self._encode_cache:
            target_vec = self._encode_cache[target_word]
            self._cache_hits += 1
        else:
            target_vec = self.encoder.encode([target_word])[0]
            self._encode_cache[target_word] = target_vec
            self._cache_misses += 1
        _encode_time = time.time() - _encode_start
        
        scores = []
        _cos_start = time.time()
        for ab in candidates:
            idx = self.word_to_idx[ab]
            sem_sim = self._cosine_sim(target_vec, self.abnormal_vectors[idx])
            pin_sim = self._pinyin_similarity(target_word, ab)
            combined = alpha * pin_sim + (1 - alpha) * sem_sim
            scores.append((ab, combined))
        _cos_time = time.time() - _cos_start
        
        _sort_start = time.time()
        scores.sort(key=lambda x: x[1], reverse=True)
        _sort_time = time.time() - _sort_start
        
        _total_time = time.time() - _start
        
        _asr_global_stats["encode_calls"] += 1
        _asr_global_stats["encode_total_time"] += _encode_time
        _asr_global_stats["find_best_calls"] += 1
        _asr_global_stats["find_best_total_time"] += _total_time
        _asr_global_stats["similarity_total_time"] += _cos_time
        
        if _asr_global_stats["find_best_calls"] % 100 == 0:
            print(f"[ASR STATS] 累计调用 {_asr_global_stats['find_best_calls']} 次")
            print(f"           encode总耗时: {_asr_global_stats['encode_total_time']:.2f}s")
            print(f"           平均encode耗时: {_asr_global_stats['encode_total_time']/_asr_global_stats['encode_calls']:.4f}s")
            print(f"           find_best总耗时: {_asr_global_stats['find_best_total_time']:.2f}s")
        
        print(f"[ASR TIMING] find_best_abnormals('{target_word}'):")
        print(f"  - encode: {_encode_time:.4f}s")
        print(f"  - similarity ({len(candidates)} candidates): {_cos_time:.4f}s")
        print(f"  - sort: {_sort_time:.4f}s")
        print(f"  - total: {_total_time:.4f}s")
        
        return [ab for ab, _ in scores[:top_k]]


def print_asr_global_stats(augmenter=None):
    """打印 ASR 噪声增强的全局统计信息"""
    stats = _asr_global_stats
    if stats["find_best_calls"] == 0:
        print("[ASR GLOBAL STATS] 无调用数据")
        return
    
    print("\n" + "="*70)
    print("[ASR GLOBAL STATS] 全局统计汇总")
    print("="*70)
    print(f"find_best_abnormals 调用次数: {stats['find_best_calls']}")
    print(f"encoder.encode 调用次数: {stats['encode_calls']}")
    print("-"*70)
    print(f"find_best_abnormals 总耗时: {stats['find_best_total_time']:.2f}s")
    print(f"encoder.encode 总耗时: {stats['encode_total_time']:.2f}s")
    print(f"相似度计算总耗时: {stats['similarity_total_time']:.2f}s")
    print("-"*70)
    print(f"平均每次 find_best_abnormals: {stats['find_best_total_time']/stats['find_best_calls']:.4f}s")
    print(f"平均每次 encode: {stats['encode_total_time']/stats['encode_calls']:.4f}s")
    print(f"encode 占比: {stats['encode_total_time']/stats['find_best_total_time']*100:.1f}%")
    if augmenter is not None:
        total = augmenter._cache_hits + augmenter._cache_misses
        if total > 0:
            print("-"*70)
            print(f"向量缓存命中: {augmenter._cache_hits} / {total} ({augmenter._cache_hits/total*100:.1f}%)")
            print(f"缓存未命中（实际encode）: {augmenter._cache_misses}")
            print(f"节省的encode调用: {augmenter._cache_hits} 次")
            saved_time = augmenter._cache_hits * (stats['encode_total_time'] / max(stats['encode_calls'], 1))
            print(f"估算节省时间: {saved_time:.2f}s")
    print("="*70 + "\n")