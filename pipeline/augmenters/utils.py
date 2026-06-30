"""
通用工具函数：切句、分词、加载词典等
"""

import logging
import os
import re
import random
from pathlib import Path
from typing import List, Set, Dict

_JIEBA_LOADED = False


def _ensure_jieba():
    global _JIEBA_LOADED
    if not _JIEBA_LOADED:
        jieba_logger = logging.getLogger("jieba")
        jieba_logger.setLevel(logging.WARNING)
        os.environ.setdefault("JIEBA_LOG_LEVEL", "WARNING")
        import jieba

        jieba.setLogLevel(logging.WARNING)
        _JIEBA_LOADED = True
    return _JIEBA_LOADED


def split_sentences(text: str) -> List[str]:
    """
    将文本按句子分割（中文句号、问号、感叹号、换行）。
    保留标点符号。
    """
    if not text:
        return []
    # 使用正则分割，保留分隔符
    pattern = r"([。！？!?．\.\n]+)"
    parts = re.split(pattern, text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sent = parts[i] + parts[i + 1]
        if sent.strip():
            sentences.append(sent)
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1])
    # 如果分割后为空，尝试按句号、问号、感叹号分割（不保留分隔符）
    if not sentences:
        sentences = [
            s.strip() for s in re.split(r"[。！？!?．\.\n]+", text) if s.strip()
        ]
    return sentences


def tokenize(text: str) -> List[str]:
    """使用 jieba 分词"""
    _ensure_jieba()
    import jieba

    return list(jieba.cut(text))


def load_word_set(file_path: str) -> Set[str]:
    """加载词典文件（每行一个词）"""
    path = Path(file_path)
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def load_synonym_dict(file_path: str) -> Dict[str, List[str]]:
    """
    加载同义词词典，支持两种格式：
      1) tab 分隔：word\\tsynonym1,synonym2,synonym3
      2) 等号分隔：code= word1 word2 word3  （每个词互为同义词）
    返回 {word: [synonym1, synonym2, ...]}
    """
    path = Path(file_path)
    if not path.exists():
        return {}
    synonym_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                parts = line.split("\t")
                if len(parts) >= 2:
                    word = parts[0].strip()
                    synonyms = [s.strip() for s in parts[1].split(",") if s.strip()]
                    if word and synonyms:
                        synonym_dict[word] = synonyms
            elif "=" in line:
                parts = line.split("=", 1)
                if len(parts) >= 2:
                    words = [w.strip() for w in parts[1].split() if w.strip()]
                    for i, w in enumerate(words):
                        others = [x for j, x in enumerate(words) if j != i]
                        if others:
                            synonym_dict[w] = others
    return synonym_dict


def load_homophone_dict(file_path: str) -> Dict[str, List[str]]:
    """加载同音词词典（格式同同义词）"""
    return load_synonym_dict(file_path)
