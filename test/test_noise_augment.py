#!/usr/bin/env python3
"""
测试词表噪声增强效果
用法:
    python test_noise_augment.py --csv <词表文件> --text "一句话"
    python test_noise_augment.py --csv <词表文件> --sentences "句子1" "句子2" "句子3"
    python test_noise_augment.py --csv <词表文件> --input input.txt --output output.txt
    python test_noise_augment.py --csv <词表文件> --text "我的账号" --replace_prob 0.5 --insert
"""

import random
import re
import argparse
from pathlib import Path
import pandas as pd
import jieba

# ---------- NoiseAugmenter 类（与 common/augment_with_noise.py 保持一致） ----------
def parse_abnormal_words(abnormal_str: str):
    """解析 '章(0.667) 张(0.333)' 格式"""
    items = abnormal_str.split()
    result = []
    for item in items:
        match = re.match(r'(.+)\(([0-9.]+)\)', item)
        if match:
            word = match.group(1)
            prob = float(match.group(2))
            result.append((word, prob))
    return result

class NoiseAugmenter:
    def __init__(self, csv_path: str, replace_prob: float = 0.3, use_insert: bool = False):
        self.replace_prob = replace_prob
        self.use_insert = use_insert
        self.patterns = {}
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            prev = row['prev_word']
            abnormal_list = parse_abnormal_words(row['abnormal_words'])
            if abnormal_list:
                self.patterns[prev] = abnormal_list

    def augment_sentence(self, sentence: str, seed: int = None):
        if seed is not None:
            random.seed(seed)
        words = list(jieba.cut(sentence))
        new_words = []
        i = 0
        while i < len(words):
            current = words[i]
            if current in self.patterns and i + 1 < len(words):
                if random.random() < self.replace_prob:
                    candidates = self.patterns[current]
                    abnormal_words = [w for w, p in candidates]
                    probs = [p for w, p in candidates]
                    selected = random.choices(abnormal_words, weights=probs, k=1)[0]
                    if self.use_insert:
                        new_words.append(current)
                        new_words.append(selected)
                        i += 1
                    else:
                        new_words.append(current)
                        new_words.append(selected)
                        i += 2
                    continue
            new_words.append(current)
            i += 1
        return ''.join(new_words)

# ---------- 测试主函数 ----------
def main():
    parser = argparse.ArgumentParser(description="测试词表噪声增强效果")
    parser.add_argument("--csv", required=True, help="prev_clean_summary.csv 文件路径")
    parser.add_argument("--text", type=str, help="待增强的单句（与 --sentences 互斥）")
    parser.add_argument("--sentences", nargs='+', help="多个句子，空格分隔")
    parser.add_argument("--input", type=str, help="输入文件（每行一句）")
    parser.add_argument("--output", type=str, help="输出文件（与 --input 搭配）")
    parser.add_argument("--replace_prob", type=float, default=0.3, help="替换概率")
    parser.add_argument("--insert", action="store_true", help="插入模式（默认替换）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    # 加载增强器
    augmenter = NoiseAugmenter(args.csv, args.replace_prob, args.insert)

    if args.sentences:
        # 多个句子模式
        for i, sent in enumerate(args.sentences):
            original = sent.strip()
            if not original:
                continue
            augmented = augmenter.augment_sentence(original, args.seed)
            print(f"[{i+1}] 原始: {original}")
            print(f"    增强: {augmented}\n")
    elif args.text:
        # 单句模式
        original = args.text.strip()
        augmented = augmenter.augment_sentence(original, args.seed)
        print(f"原始句子: {original}")
        print(f"增强句子: {augmented}")
    elif args.input:
        # 文件模式
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"错误：输入文件不存在 {input_path}")
            return
        output_path = Path(args.output) if args.output else input_path.with_suffix(".aug.txt")
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                line = line.strip()
                if line:
                    aug_line = augmenter.augment_sentence(line, args.seed)
                    fout.write(aug_line + '\n')
        print(f"增强完成，结果保存至: {output_path}")
    else:
        print("请提供 --text 或 --sentences 或 --input")

if __name__ == "__main__":
    main()