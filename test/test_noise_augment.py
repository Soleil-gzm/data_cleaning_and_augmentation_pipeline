#!/usr/bin/env python3
"""
硬编码测试脚本：验证 ASR 噪声增强器效果
使用前请修改下方的四个路径变量
"""
import sys
import random
import jieba
import re
from pathlib import Path

# 添加项目根目录到 Python 路径（假设脚本放在根目录或 test/ 下）
sys.path.insert(0, str(Path(__file__).parent))
from common.asr_noise_augmenter import AsrNoiseAugmenter

# ========== 请修改以下路径 ==========
MODEL_PATH = "Models/paraphrase-multilingual-MiniLM-L12-v2"
VECTORS_PATH = "resources/prev_clean/sample_20/qwen/prev_clean_prev_window_2_no_prob/abnormal_vectors.pkl"
PINYIN_PATH = "resources/prev_clean/sample_20/qwen/prev_clean_prev_window_2_no_prob/abnormal_pinyin.pkl"
PREV_MAP_PATH = "resources/prev_clean/sample_20/qwen/prev_clean_prev_window_2_no_prob/prev_to_abnormals.pkl"
# ===================================

# 可选：调整拼音权重（0~1，越大越偏向拼音）
ALPHA = 0.7
# 随机种子（便于复现）
RANDOM_SEED = 42

def apply_asr_noise(sentence: str, augmenter: AsrNoiseAugmenter) -> str:
    """对句子中随机一个中文词语进行 ASR 噪声替换"""
    if not sentence or not sentence.strip():
        return sentence
    words = jieba.lcut(sentence)
    if not words:
        return sentence
    # 选择中文字词（仅中文，不含标点）
    candidates_idx = [i for i, w in enumerate(words) if re.fullmatch(r'[\u4e00-\u9fa5]+', w)]
    if not candidates_idx:
        return sentence
    idx = random.choice(candidates_idx)
    target = words[idx]
    candidates = augmenter.find_best_abnormals(target, top_k=5, alpha=ALPHA)
    if candidates:
        chosen = random.choice(candidates)
        words[idx] = chosen
    return ''.join(words)

def main():
    # 设置随机种子
    random.seed(RANDOM_SEED)
    jieba.initialize()

    print("=" * 70)
    print("ASR 噪声增强器测试（硬编码版本）")
    print("=" * 70)

    # 1. 检查文件是否存在
    for path, name in [(VECTORS_PATH, "向量文件"), (PINYIN_PATH, "拼音文件"), (MODEL_PATH, "模型文件夹")]:
        if not Path(path).exists():
            print(f"错误：{name} 不存在: {path}")
            sys.exit(1)
    if PREV_MAP_PATH and not Path(PREV_MAP_PATH).exists():
        print(f"警告：前置词映射文件不存在，将不使用前置词限制: {PREV_MAP_PATH}")

    # 2. 加载增强器
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

    # 3. 测试句子集
    test_sentences = [
        "我的信用卡逾期了，怎么办？",
        "请尽快还款，否则会影响征信。",
        "这个月账单我已经还清了。",
        "客服态度很好，帮我处理了问题。",
        "我需要申请分期还款。"
    ]

    print("【句子级别增强测试】每个原句生成 3 个变体")
    print("-" * 70)
    for original in test_sentences:
        variants = []
        for _ in range(3):
            var = apply_asr_noise(original, augmenter)
            variants.append(var)
        print(f"原句: {original}")
        for i, v in enumerate(variants, 1):
            print(f"  变体{i}: {v}")
        print()

    # 4. 单词语义+拼音匹配演示
    print("【单词语义+拼音匹配演示】显示每个词的前5个候选异常词")
    print("-" * 70)
    demo_words = ["逾期", "还款", "征信", "客服", "银行"]
    for w in demo_words:
        cand = augmenter.find_best_abnormals(w, top_k=5, alpha=ALPHA)
        print(f"{w} -> {cand}")

    print("\n测试完成。")

if __name__ == "__main__":
    main()