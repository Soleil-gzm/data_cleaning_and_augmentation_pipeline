#!/usr/bin/env python3
"""
基于前置词-异常词词表的数据增强工具
策略：匹配前置词后，以一定概率将后面的词替换为异常词（或插入异常词）
支持：
   - 替换模式（默认）或插入模式
   - 自定义替换概率
   - 随机种子
用法：
    python augment_with_noise.py --csv <词表.csv> --input <输入.txt> --output <输出.txt> [--replace_prob 0.3] [--insert] [--seed 42]
"""

import random
import re
import argparse
from pathlib import Path
import pandas as pd
import jieba

def parse_abnormal_words(abnormal_str: str):
    """
    解析形如 '章(0.667) 张(0.333)' 的字符串，返回 (word, prob) 列表
    """
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
        self.patterns = {}   # {prev_word: [(abnormal_word, prob), ...]}
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            prev = row['prev_word']
            abnormal_list = parse_abnormal_words(row['abnormal_words'])
            if abnormal_list:
                self.patterns[prev] = abnormal_list

    def augment_sentence(self, sentence: str, seed: int = None):
        if seed is not None:
            random.seed(seed)
        # 使用 jieba 分词
        words = list(jieba.cut(sentence))
        new_words = []
        i = 0
        while i < len(words):
            current = words[i]
            # 如果当前词是前置词，且后面还有词
            if current in self.patterns and i + 1 < len(words):
                if random.random() < self.replace_prob:
                    # 按概率选择异常词
                    candidates = self.patterns[current]
                    abnormal_words = [w for w, p in candidates]
                    probs = [p for w, p in candidates]
                    selected = random.choices(abnormal_words, weights=probs, k=1)[0]
                    if self.use_insert:
                        # 插入模式：保留当前词，然后插入异常词
                        new_words.append(current)
                        new_words.append(selected)
                        i += 1   # 移动一个词（当前词已处理，下一个词是原来的下一个）
                    else:
                        # 替换模式：保留当前词，替换下一个词为异常词
                        new_words.append(current)
                        new_words.append(selected)
                        i += 2   # 跳过当前词和下一个词
                    continue
            new_words.append(current)
            i += 1
        # 将词列表合并为字符串（中文无需空格）
        return ''.join(new_words)

    def augment_file(self, input_path: Path, output_path: Path, seed: int = 42):
        random.seed(seed)
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                line = line.strip()
                if line:
                    aug_line = self.augment_sentence(line)
                    fout.write(aug_line + '\n')
                else:
                    fout.write('\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="prev_clean_summary.csv 文件路径")
    parser.add_argument("--input", required=True, help="输入文本文件（每行一句）")
    parser.add_argument("--output", required=True, help="输出增强文本文件")
    parser.add_argument("--replace_prob", type=float, default=0.3, help="替换概率")
    parser.add_argument("--insert", action="store_true", help="使用插入模式（默认替换）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    augmenter = NoiseAugmenter(args.csv, args.replace_prob, args.insert)
    augmenter.augment_file(Path(args.input), Path(args.output), args.seed)
    print(f"增强完成，输出文件：{args.output}")

if __name__ == "__main__":
    main()