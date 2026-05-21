#!/usr/bin/env python3
"""
测试 ASR 噪声增强器
用法：python test_asr_augmenter.py --vectors_path resources/prev_clean/.../abnormal_vectors.pkl --pinyin_path resources/prev_clean/.../abnormal_pinyin.pkl
（路径请替换为实际生成的 .pkl 文件所在位置）
"""

import argparse
import random
import jieba
import re
from pathlib import Path
import sys

# 添加项目路径以便导入 common 模块
sys.path.insert(0, str(Path(__file__).parent))

from common.asr_noise_augmenter import AsrNoiseAugmenter

def apply_asr_noise(sentence: str, augmenter: AsrNoiseAugmenter, alpha=0.7, top_k=5) -> str:
    """
    对句子中随机一个中文词语进行 ASR 噪声替换
    """
    if not sentence or not sentence.strip():
        return sentence
    words = jieba.lcut(sentence)
    if not words:
        return sentence
    # 找出所有中文字词（长度>=1且全是中文）
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
    parser = argparse.ArgumentParser(description="测试 ASR 噪声增强器")
    parser.add_argument("--vectors_path", required=True, help="abnormal_vectors.pkl 文件路径")
    parser.add_argument("--pinyin_path", required=True, help="abnormal_pinyin.pkl 文件路径")
    parser.add_argument("--prev_map_path", default=None, help="prev_to_abnormals.pkl 文件路径（可选）")
    parser.add_argument("--alpha", type=float, default=0.7, help="拼音权重（0~1）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)
    jieba.initialize()

    # 加载增强器
    print("正在加载 ASR 增强器...")
    augmenter = AsrNoiseAugmenter(
        vectors_path=args.vectors_path,
        pinyin_path=args.pinyin_path,
        prev_map_path=args.prev_map_path
    )
    print(f"异常词数量: {len(augmenter.abnormal_words)}")
    print("加载完成。\n")

    # 测试句子（您可以根据需要修改）
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
        # 生成 3 个变体（每次随机选择不同词语替换）
        variants = []
        for _ in range(3):
            variant = apply_asr_noise(original, augmenter, alpha=args.alpha)
            variants.append(variant)
        print(f"原句: {original}")
        for i, v in enumerate(variants, 1):
            print(f"  变体{i}: {v}")
        print()

    # 可选：演示对单个词的查找
    print("\n[演示] 对单个词的候选异常词（top-5）：")
    demo_words = ["逾期", "还款", "征信", "客服"]
    for w in demo_words:
        candidates = augmenter.find_best_abnormals(w, top_k=5, alpha=args.alpha)
        print(f"{w} -> {candidates}")

if __name__ == "__main__":
    main()