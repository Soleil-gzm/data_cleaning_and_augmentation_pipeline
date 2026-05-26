#!/usr/bin/env python3
"""
测试 ASR 噪声增强器（硬编码路径版）- 带详细内部日志
用法：直接修改下方的路径变量，然后运行。
"""
import sys
import random
import jieba
import re
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from common import augment_utils_add as aug_utils
from common.asr_noise_augmenter import AsrNoiseAugmenter

# ========== 请根据实际路径修改 ==========
MODEL_PATH = "Models/paraphrase-multilingual-MiniLM-L12-v2"
VECTORS_PATH = "resources/prev_clean/sample_20/qwen/prev_clean_prev_window_1_no_prob/abnormal_vectors.pkl"
PINYIN_PATH = "resources/prev_clean/sample_20/qwen/prev_clean_prev_window_1_no_prob/abnormal_pinyin.pkl"
PREV_MAP_PATH = "resources/prev_clean/sample_20/qwen/prev_clean_prev_window_1_no_prob/prev_to_abnormals.pkl"
# ========================================

# 增强参数
ALPHA = 0.7
MAX_OPERATIONS = 2
INSERT_PROB = 0.2
RETRY_TIMES = 3
RANDOM_SEED = 42

# 肯定/否定词集（用于极性检测）
AFFIRMATIVE_WORDS = {"是", "有", "能", "可以", "行", "好", "对", "是的", "没错", "肯定", "必须", "需要", "会", "应该"}
NEGATIVE_WORDS = {"不", "没", "无", "别", "不要", "不用", "不行", "不是", "没有", "不能", "不可以", "否定", "不会", "不该"}

def create_debug_apply_asr_noise(augmenter):
    """返回一个带详细打印的 apply_asr_noise 函数，使用给定的 augmenter"""
    def debug_apply_asr_noise(sentence: str) -> str:
        print(f"\n{'='*60}")
        print(f"[DEBUG] 输入句子: {sentence}")
        if augmenter is None:
            print("[DEBUG] 增强器为 None，返回原句")
            return sentence
        if not sentence or not sentence.strip():
            return sentence

        # 局部增强函数（复制自原函数，但加入打印）
        def enhance_once(sent):
            words = jieba.lcut(sent)
            print(f"[DEBUG] 分词结果: {words}")
            if len(words) < 2:
                return sent

            # 找出可操作位置
            candidate_indices = []
            for i in range(1, len(words)):
                if words[i-1] in augmenter.prev_to_abnormals:
                    candidate_indices.append(i)
            print(f"[DEBUG] 可操作的目标词索引（基于前置词匹配）: {candidate_indices}")
            if not candidate_indices:
                return sent

            max_ops = min(MAX_OPERATIONS, len(candidate_indices))
            selected = []
            shuffled = random.sample(candidate_indices, len(candidate_indices))
            for idx in shuffled:
                if not selected or all(abs(idx - x) >= 2 for x in selected):
                    selected.append(idx)
                    if len(selected) >= max_ops:
                        break
            print(f"[DEBUG] 选中的位置（不重叠）: {selected}")

            operations = []
            for pos in selected:
                prev_word = words[pos-1]
                target_word = words[pos]
                print(f"\n  --- 处理位置 {pos}: 前置词='{prev_word}', 目标词='{target_word}' ---")
                # 获取候选异常词
                candidates = augmenter.find_best_abnormals(
                    target_word, prev_word=prev_word, top_k=5, alpha=ALPHA
                )
                print(f"  [候选异常词] (top-{len(candidates)}): {candidates}")
                if not candidates:
                    print("  无候选词，跳过此位置")
                    continue

                # 极性检测
                target_polarity = None
                if target_word in AFFIRMATIVE_WORDS:
                    target_polarity = "affirmative"
                elif target_word in NEGATIVE_WORDS:
                    target_polarity = "negative"
                print(f"  目标词极性: {target_polarity}")

                chosen = None
                for _ in range(5):
                    cand = random.choice(candidates)
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
                    print("  经过极性检测后，没有找到合适的候选词，跳过")
                    continue

                # 决定操作类型
                is_insert = random.random() < INSERT_PROB
                op_type = "插入" if is_insert else "替换"
                print(f"  选中异常词: '{chosen}' (操作: {op_type})")
                operations.append((pos, chosen, is_insert))

            if not operations:
                print("[DEBUG] 没有有效操作，返回原句")
                return sent

            # 应用操作
            new_words = words[:]
            for pos, new_word, is_insert in sorted(operations, key=lambda x: x[0], reverse=True):
                if is_insert:
                    new_words.insert(pos, new_word)
                    print(f"  [应用] 在位置 {pos} 之前插入 '{new_word}'")
                else:
                    new_words[pos] = new_word
                    print(f"  [应用] 替换位置 {pos} 的 '{words[pos]}' 为 '{new_word}'")
            result = ''.join(new_words)
            print(f"[DEBUG] 增强后句子: {result}")
            return result

        original = sentence
        for attempt in range(RETRY_TIMES):
            print(f"\n[重试次数: {attempt+1}/{RETRY_TIMES}]")
            result = enhance_once(original)
            if result != original:
                print(f"[最终结果] 成功生成变体: {result}")
                return result
        print("[最终结果] 多次尝试后仍无变化，返回原句")
        return original
    return debug_apply_asr_noise

def main():
    random.seed(RANDOM_SEED)
    jieba.initialize()

    print("="*70)
    print("ASR 噪声增强器详细测试（含内部候选词得分）")
    print("="*70)

    # 加载增强器
    print("正在加载 ASR 增强器...")
    augmenter = AsrNoiseAugmenter(
        vectors_path=VECTORS_PATH,
        pinyin_path=PINYIN_PATH,
        prev_map_path=PREV_MAP_PATH,
        model_path=MODEL_PATH
    )
    print(f"异常词数量: {len(augmenter.abnormal_words)}")
    print(f"前置词映射大小: {len(augmenter.prev_to_abnormals)}")
    print("加载完成。\n")

    # 临时替换全局增强器
    aug_utils.set_asr_augmenter(augmenter)
    # 用带打印的函数覆盖
    aug_utils.apply_asr_noise = create_debug_apply_asr_noise(augmenter)

    # 测试句子
    test_sentences = [
        "我的信用卡逾期了，怎么办？",
        "请尽快还款，否则会影响征信。",
        "我能处理这个问题。",
        "你不能这样做。"
    ]

    for sent in test_sentences:
        print("\n" + "="*70)
        print(f"测试句子: {sent}")
        for i in range(2):
            print(f"\n--- 生成变体 {i+1} ---")
            variant = aug_utils.apply_asr_noise(sent)
            print(f"变体结果: {variant}")
        print("="*70)

if __name__ == "__main__":
    main()