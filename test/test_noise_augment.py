#!/usr/bin/env python3
"""
硬编码测试脚本：ASR 噪声增强器（前置词匹配 + 多操作 + 插入）
使用前请修改下方的路径变量
"""
import sys
import random
import jieba
import re
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent))
from common.asr_noise_augmenter import AsrNoiseAugmenter

# ========== 请修改以下路径 ==========
MODEL_PATH = "Models/paraphrase-multilingual-MiniLM-L12-v2"
VECTORS_PATH = "resources/prev_clean/sample_20/qwen/prev_clean_prev_window_2_no_prob/abnormal_vectors.pkl"
PINYIN_PATH = "resources/prev_clean/sample_20/qwen/prev_clean_prev_window_2_no_prob/abnormal_pinyin.pkl"
PREV_MAP_PATH = "resources/prev_clean/sample_20/qwen/prev_clean_prev_window_2_no_prob/prev_to_abnormals.pkl"
# ===================================

# 增强参数
ALPHA = 0.7                     # 拼音权重
RANDOM_SEED = 42
MAX_OPERATIONS = 2              # 每个句子最多进行几次替换/插入操作（避免过度增强）
INSERT_PROB = 0.1               # 插入操作的概率（否则为替换）
print("插入概率：",INSERT_PROB)
RETRY_TIMES = 3                 # 若变体与原句相同，最多重试次数

def enhance_sentence(sentence: str, augmenter: AsrNoiseAugmenter) -> str:
    """
    对句子进行 ASR 噪声增强（支持替换/插入，利用前置词匹配）
    返回增强后的句子。
    """
    if not sentence or not sentence.strip():
        return sentence

    # 分词（保留原始分隔符信息，简单用 jieba 分词）
    words = jieba.lcut(sentence)
    if len(words) <= 1:
        return sentence

    # 决定进行多少次操作（1 到 MAX_OPERATIONS 之间随机）
    num_ops = random.randint(1, MAX_OPERATIONS)
    # 为了避免重复操作同一位置，记录已操作过的索引（替换后词语长度可能变化，这里简单用位置索引+偏移处理较复杂，我们采用重新扫描的方式）
    # 更简单的方法：每次操作后重新分词，并对新词序列再次操作（但可能导致指数增长）
    # 为了简化，我们每操作一次就重新生成一次句子，循环 num_ops 次
    current_sentence = sentence
    for _ in range(num_ops):
        # 对当前句子重新分词
        cur_words = jieba.lcut(current_sentence)
        if len(cur_words) < 2:
            break
        # 寻找可以操作的位置（需要前置词匹配）
        # 遍历词列表，对于位置 i（从1开始），检查 words[i-1] 是否在 prev_to_abnormals 中
        candidates_pos = []
        for i in range(1, len(cur_words)):
            prev_word = cur_words[i-1]
            if prev_word in augmenter.prev_to_abnormals:
                candidates_pos.append(i)   # 可以操作当前词（cur_words[i]）
        if not candidates_pos:
            # 没有可操作的位置，退出
            break
        # 随机选择一个位置
        pos = random.choice(candidates_pos)
        prev_word = cur_words[pos-1]
        target_word = cur_words[pos]
        # 获取候选异常词（基于前置词限制）
        candidates = augmenter.find_best_abnormals(
            target_word, 
            prev_word=prev_word, 
            top_k=5, 
            alpha=ALPHA
        )
        if not candidates:
            continue
        # 随机选择是否插入（INSERT_PROB）还是替换
        if random.random() < INSERT_PROB:
            # 插入：在前置词之后、目标词之前插入一个异常词
            insert_word = random.choice(candidates)
            new_words = cur_words[:pos] + [insert_word] + cur_words[pos:]
        else:
            # 替换：用异常词替换目标词
            replace_word = random.choice(candidates)
            new_words = cur_words[:pos] + [replace_word] + cur_words[pos+1:]
        # 重新拼接成字符串
        current_sentence = ''.join(new_words)
    return current_sentence

def apply_asr_noise_with_retry(sentence: str, augmenter: AsrNoiseAugmenter) -> str:
    """
    包装增强函数，如果结果与原句相同则重试，直到不同或达到最大重试次数
    """
    original = sentence
    for attempt in range(RETRY_TIMES):
        new_sent = enhance_sentence(original, augmenter)
        if new_sent != original:
            return new_sent
    # 如果始终相同，返回原句（不做修改）
    return original

def main():
    random.seed(RANDOM_SEED)
    jieba.initialize()

    print("=" * 70)
    print("ASR 噪声增强器测试（前置词匹配 + 多操作 + 插入）")
    print("=" * 70)

    # 检查文件
    for path, name in [(VECTORS_PATH, "向量文件"), (PINYIN_PATH, "拼音文件"), (MODEL_PATH, "模型文件夹")]:
        if not Path(path).exists():
            print(f"错误：{name} 不存在: {path}")
            sys.exit(1)
    if not Path(PREV_MAP_PATH).exists():
        print(f"警告：前置词映射文件不存在，将无法使用前置词限制: {PREV_MAP_PATH}")

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

    # 测试句子
    test_sentences = [
        "我的信用卡逾期了，怎么办？",
        "请尽快还款，否则会影响征信。",
        "这个月账单我已经还清了。",
        "客服态度很好，帮我处理了问题。",
        "我需要申请分期还款。"
    ]

    print("【句子增强测试】每个原句生成 3 个变体（支持多操作、插入）")
    print("-" * 70)
    for original in test_sentences:
        variants = []
        for _ in range(3):
            variant = apply_asr_noise_with_retry(original, augmenter)
            variants.append(variant)
        print(f"原句: {original}")
        for i, v in enumerate(variants, 1):
            print(f"  变体{i}: {v}")
        print()

    # 可选：展示前置词映射样例
    print("【前置词映射示例】")
    sample_prev = list(augmenter.prev_to_abnormals.keys())[:5]
    for prev in sample_prev:
        abnormals = augmenter.prev_to_abnormals[prev][:10]  # 只显示前10个
        print(f"{prev} -> {abnormals}")

if __name__ == "__main__":
    main()