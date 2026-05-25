#!/usr/bin/env python3
"""
测试 ASR 噪声增强的极性保护功能
验证肯定词不会被替换成否定词，反之亦然。
"""
import sys
import random
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from common.asr_noise_augmenter import AsrNoiseAugmenter
from common import augment_utils_add as aug_utils

# ========== 配置（请修改为您的实际路径）==========
MODEL_PATH = "Models/paraphrase-multilingual-MiniLM-L12-v2"
VECTORS_PATH = "resources/prev_clean/sample_20/qwen/prev_clean_prev_window_3_no_prob/abnormal_vectors.pkl"
PINYIN_PATH = "resources/prev_clean/sample_20/qwen/prev_clean_prev_window_3_no_prob/abnormal_pinyin.pkl"
PREV_MAP_PATH = "resources/prev_clean/sample_20/qwen/prev_clean_prev_window_3_no_prob/prev_to_abnormals.pkl"
# ================================================

# 肯定词和否定词（用于检查）
AFFIRMATIVE_WORDS = {"是", "有", "能", "可以", "行", "好", "对", "是的", "没错", "肯定", "必须", "需要", "会", "应该"}
NEGATIVE_WORDS = {"不", "没", "无", "别", "不要", "不用", "不行", "不是", "没有", "不能", "不可以", "否定", "不会", "不该"}

def check_polarity_preserved(original_sent, new_sent):
    """检查新句子是否保留了原句中的肯定/否定词极性（简单检测）"""
    # 找出原句中出现的肯定/否定词
    orig_aff = [w for w in AFFIRMATIVE_WORDS if w in original_sent]
    orig_neg = [w for w in NEGATIVE_WORDS if w in original_sent]
    # 如果原句没有极性词，直接认为通过
    if not orig_aff and not orig_neg:
        return True, "无极性词"
    # 检查新句子中是否出现了相反的极性词
    new_aff = [w for w in AFFIRMATIVE_WORDS if w in new_sent]
    new_neg = [w for w in NEGATIVE_WORDS if w in new_sent]
    # 如果原句有肯定词，新句没有肯定词却新增了否定词 -> 异常
    if orig_aff and not new_aff and new_neg:
        return False, f"肯定词 {orig_aff} 被替换为否定词 {new_neg}"
    # 如果原句有否定词，新句没有否定词却新增了肯定词 -> 异常
    if orig_neg and not new_neg and new_aff:
        return False, f"否定词 {orig_neg} 被替换为肯定词 {new_aff}"
    return True, "极性一致"

def main():
    print("="*70)
    print("测试 ASR 噪声增强的极性保护功能")
    print("="*70)

    # 加载增强器
    print("正在加载 ASR 增强器...")
    try:
        augmenter = AsrNoiseAugmenter(
            vectors_path=VECTORS_PATH,
            pinyin_path=PINYIN_PATH,
            prev_map_path=PREV_MAP_PATH,
            model_path=MODEL_PATH
        )
        aug_utils.set_asr_augmenter(augmenter)
        print(f"异常词数量: {len(augmenter.abnormal_words)}")
        print(f"前置词映射大小: {len(augmenter.prev_to_abnormals)}")
    except Exception as e:
        print(f"加载增强器失败: {e}")
        sys.exit(1)

    # 测试句子（包含肯定/否定词）
    test_sentences = [
        "我能处理这个问题。",          # 肯定词“能”
        "你不能这样做。",              # 否定词“不能”
        "我会按时还款。",              # 肯定词“会”
        "他不是一个好人。",            # 否定词“不是”
        "请尽快还款，否则会影响征信。", # 无极性词（对照）
        "我的信用卡逾期了，怎么办？",   # 无极性词
        "你应该还钱。",                # 肯定词“应该”
        "不要逾期。",                  # 否定词“不要”
    ]

    print("\n开始测试（每个句子生成 3 个变体）...")
    print("-"*70)
    all_pass = True
    for sent in test_sentences:
        print(f"原句: {sent}")
        variants = []
        for i in range(3):
            variant = aug_utils.apply_asr_noise(sent)
            variants.append(variant)
            ok, msg = check_polarity_preserved(sent, variant)
            if not ok:
                print(f"  ❌ 变体{i+1}: {variant}  [{msg}]")
                all_pass = False
            else:
                print(f"  ✅ 变体{i+1}: {variant}")
        print()

    if all_pass:
        print("🎉 所有变体均通过极性测试（未发现肯定/否定词翻转）")
    else:
        print("⚠️ 存在极性翻转错误，请检查极性检测逻辑或词表。")

if __name__ == "__main__":
    main()