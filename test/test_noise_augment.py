#!/usr/bin/env python3
"""
ASR 噪声增强器完整测试脚本（硬编码路径版）
功能：
1. 一次性扫描句子，找出所有符合前置词条件的目标词位置
2. 随机选择最多 MAX_OPERATIONS 个互不相邻的位置
3. 对每个选中位置独立决定替换或插入（概率由 INSERT_PROB 控制）
4. 从异常词候选中随机均匀选择一个进行替换/插入（可扩展为按频率加权）
5. 从后往前应用所有修改，避免索引偏移
6. 若增强后句子与原句相同，则重试最多 RETRY_TIMES 次

使用方法：
- 修改下方的 MODEL_PATH, VECTORS_PATH, PINYIN_PATH, PREV_MAP_PATH
- python test_asr_hardcoded.py
"""
import sys
import random
import jieba
import re
from pathlib import Path

# 添加项目根目录到路径（假设脚本放在项目根目录或 test/ 下）
sys.path.insert(0, str(Path(__file__).parent))
from common.asr_noise_augmenter import AsrNoiseAugmenter

# ========== 请根据实际路径修改 ==========
MODEL_PATH = "Models/paraphrase-multilingual-MiniLM-L12-v2"
VECTORS_PATH = "resources/prev_clean/sample_20/qwen/prev_clean_prev_window_1_no_prob/abnormal_vectors.pkl"
PINYIN_PATH = "resources/prev_clean/sample_20/qwen/prev_clean_prev_window_1_no_prob/abnormal_pinyin.pkl"
PREV_MAP_PATH = "resources/prev_clean/sample_20/qwen/prev_clean_prev_window_1_no_prob/prev_to_abnormals.pkl"
# ======================================

# 增强参数
ALPHA = 0.7                 # 拼音权重（0~1，越大越偏向拼音）
MAX_OPERATIONS = 2          # 每个句子最多执行的操作次数（替换或插入）
INSERT_PROB = 0.1           # 插入操作的概率（否则为替换）
RETRY_TIMES = 3             # 若变体与原句相同，最多重试次数
RANDOM_SEED = 42            # 随机种子，便于复现

def enhance_sentence_once(sentence: str, augmenter: AsrNoiseAugmenter) -> str:
    """
    对单个句子进行一次增强（可能修改多个位置）
    返回增强后的句子（可能与原句相同）
    """
    if not sentence or not sentence.strip():
        return sentence
    words = jieba.lcut(sentence)
    if len(words) < 2:
        return sentence

    # 1. 找出所有可操作的目标词索引（i >= 1 且 words[i-1] 是前置词）
    candidate_indices = []
    for i in range(1, len(words)):
        if words[i-1] in augmenter.prev_to_abnormals:
            candidate_indices.append(i)

    if not candidate_indices:
        return sentence

    # 2. 随机选择最多 MAX_OPERATIONS 个互不相邻的位置（避免前置词冲突）
    max_ops = min(MAX_OPERATIONS, len(candidate_indices))
    selected = []
    # 随机打乱候选列表
    shuffled = random.sample(candidate_indices, len(candidate_indices))
    for idx in shuffled:
        if not selected or all(abs(idx - x) >= 2 for x in selected):
            selected.append(idx)
            if len(selected) >= max_ops:
                break

    # 3. 为每个选中位置生成操作（替换或插入）
    operations = []  # (pos, new_word, is_insert)
    for pos in selected:
        prev_word = words[pos-1]
        target_word = words[pos]
        # 获取候选异常词（基于前置词限制）
        candidates = augmenter.find_best_abnormals(
            target_word,
            prev_word=prev_word,
            top_k=5,
            alpha=ALPHA
        )
        if not candidates:
            continue
        chosen = random.choice(candidates)
        if random.random() < INSERT_PROB:
            # 插入：在目标词之前插入异常词，原词保留
            operations.append((pos, chosen, True))
        else:
            # 替换：用异常词替换目标词
            operations.append((pos, chosen, False))

    if not operations:
        return sentence

    # 4. 从后往前应用操作（避免索引偏移）
    new_words = words[:]
    for pos, new_word, is_insert in sorted(operations, key=lambda x: x[0], reverse=True):
        if is_insert:
            new_words.insert(pos, new_word)
        else:
            new_words[pos] = new_word
    return ''.join(new_words)

def enhance_sentence_with_retry(sentence: str, augmenter: AsrNoiseAugmenter) -> str:
    """
    包装增强函数，如果结果与原句相同则重试，直到不同或达到最大重试次数
    """
    original = sentence
    for attempt in range(RETRY_TIMES):
        new_sent = enhance_sentence_once(original, augmenter)
        if new_sent != original:
            return new_sent
    return original  # 多次尝试后仍相同则返回原句

def main():
    random.seed(RANDOM_SEED)
    jieba.initialize()

    print("=" * 70)
    print("ASR 噪声增强器测试（一次性定位多个非重叠位置 + 批量操作）")
    print("=" * 70)

    # 检查文件存在性
    for path, name in [(VECTORS_PATH, "向量文件"), (PINYIN_PATH, "拼音文件"), (MODEL_PATH, "模型文件夹")]:
        if not Path(path).exists():
            print(f"错误：{name} 不存在: {path}")
            sys.exit(1)
    if not Path(PREV_MAP_PATH).exists():
        print(f"警告：前置词映射文件不存在，将无法使用前置词限制: {PREV_MAP_PATH}")
        print("增强器会尝试加载该文件，若文件缺失则跳过前置词匹配。")

    # 加载增强器
    print("正在加载 ASR 增强器...")
    augmenter = AsrNoiseAugmenter(
        vectors_path=VECTORS_PATH,
        pinyin_path=PINYIN_PATH,
        prev_map_path=PREV_MAP_PATH if Path(PREV_MAP_PATH).exists() else None,
        model_path=MODEL_PATH
    )
    print(f"异常词数量: {len(augmenter.abnormal_words)}")
    print(f"前置词种类: {len(augmenter.prev_to_abnormals)}")
    print("加载完成。\n")

    # 测试句子集（可根据需要修改）
    test_sentences = [
        "我的信用卡逾期了，怎么办？",
        "请尽快还款，否则会影响征信。",
        "这个月账单我已经还清了。",
        "客服态度很好，帮我处理了问题。",
        "我需要申请分期还款。"
    ]

    print("【句子增强测试】每个原句生成 3 个变体")
    print("-" * 70)
    for original in test_sentences:
        variants = []
        for _ in range(3):
            variant = enhance_sentence_with_retry(original, augmenter)
            variants.append(variant)
        print(f"原句: {original}")
        for i, v in enumerate(variants, 1):
            print(f"  变体{i}: {v}")
        print()

    # 展示部分前置词映射示例
    print("【前置词映射示例】（前5个）")
    sample_prev = list(augmenter.prev_to_abnormals.keys())[:5]
    for prev in sample_prev:
        abnormals = augmenter.prev_to_abnormals[prev][:10]  # 只显示前10个
        print(f"{prev} -> {abnormals}")

    # 可选：演示单个词的候选词（方便调试）
    print("\n【单词语义+拼音匹配演示】")
    demo_words = ["逾期", "还款", "征信", "客服", "银行"]
    for w in demo_words:
        # 不指定前置词，候选集为所有异常词
        cand = augmenter.find_best_abnormals(w, top_k=5, alpha=ALPHA)
        print(f"{w} -> {cand}")

if __name__ == "__main__":
    main()