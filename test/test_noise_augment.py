#!/usr/bin/env python3
"""
测试 ASR 噪声增强器（本地模型版）
用法：python test/test_asr_augmenter.py --vectors_path ... --pinyin_path ... --model_path /path/to/local/model
"""
import argparse
import random
import jieba
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))  # 添加项目根目录
from common.asr_noise_augmenter import AsrNoiseAugmenter

def apply_asr_noise(sentence: str, augmenter: AsrNoiseAugmenter, alpha=0.7, top_k=5) -> str:
    if not sentence or not sentence.strip():
        return sentence
    words = jieba.lcut(sentence)
    if not words:
        return sentence
    candidates_idx = [i for i, w in enumerate(words) if re.fullmatch(r'[\u4e00-\u9fa5]+', w)]
    if not candidates_idx:
        return sentence
    idx = random.choice(candidates_idx)
    target = words[idx]
    abnormal_list = augmenter.find_best_abnormals(target, top_k=top_k, alpha=alpha)
    if abnormal_list:
        chosen = random.choice(abnormal_list)
        words[idx] = chosen
    return ''.join(words)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors_path", required=True)
    parser.add_argument("--pinyin_path", required=True)
    parser.add_argument("--prev_map_path", default=None)
    parser.add_argument("--model_path", required=True, help="本地 SentenceTransformer 模型文件夹路径")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    jieba.initialize()

    print("正在加载 ASR 增强器（本地模型）...")
    augmenter = AsrNoiseAugmenter(
        vectors_path=args.vectors_path,
        pinyin_path=args.pinyin_path,
        prev_map_path=args.prev_map_path,
        model_path=args.model_path
    )
    print(f"异常词数量: {len(augmenter.abnormal_words)}")
    print("加载完成。\n")

    test_sentences = [
        "我的信用卡逾期了，怎么办？",
        "请尽快还款，否则会影响征信。",
        "这个月账单我已经还清了。",
        "客服态度很好，帮我处理了问题。",
        "我需要申请分期还款。"
    ]

    print("=" * 60)
    print("原始句子 -> 增强后句子")
    print("=" * 60)

    for original in test_sentences:
        variants = []
        for _ in range(3):
            variant = apply_asr_noise(original, augmenter, alpha=args.alpha)
            variants.append(variant)
        print(f"原句: {original}")
        for i, v in enumerate(variants, 1):
            print(f"  变体{i}: {v}")
        print()

    print("\n[演示] 对单个词的候选异常词（top-5）：")
    demo_words = ["逾期", "还款", "征信", "客服"]
    for w in demo_words:
        candidates = augmenter.find_best_abnormals(w, top_k=5, alpha=args.alpha)
        print(f"{w} -> {candidates}")

if __name__ == "__main__":
    main()