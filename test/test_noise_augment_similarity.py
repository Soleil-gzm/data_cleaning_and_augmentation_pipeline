#!/usr/bin/env python3
"""
测试词表噪声增强（基于 Qwen/GPT-2 模型的向量相似度）
策略：匹配前置词后，计算下一个真实词与候选异常词的余弦相似度，选择最相似的异常词替换。
用法:
    # 单句
    python test_noise_augment_model_sim.py --model_name <模型名> --csv <词表文件> --text "一句话"
    # 多个句子（空格分隔）
    python test_noise_augment_model_sim.py --model_name <模型名> --csv <词表文件> --sentences "句子1" "句子2" "句子3"
    # 文件输入
    python test_noise_augment_model_sim.py --model_name <模型名> --csv <词表文件> --input input.txt --output output.txt
"""

import re
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import jieba
import torch
from transformers import AutoTokenizer, AutoModel

def parse_abnormal_words(abnormal_str: str):
    items = abnormal_str.split()
    words = []
    for item in items:
        match = re.match(r'(.+)\([0-9.]+\)', item)
        if match:
            words.append(match.group(1))
    return words

class ModelSimilarityAugmenter:
    def __init__(self, model_name: str, csv_path: str, device: str = None, trust_remote_code: bool = True):
        self.patterns = {}
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            prev = row['prev_word']
            abnormal_list = parse_abnormal_words(row['abnormal_words'])
            if abnormal_list:
                self.patterns[prev] = abnormal_list

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code).to(self.device)
        self.model.eval()
        # 预计算所有异常词的嵌入
        self.abnormal_embeddings = {}
        for prev, words in self.patterns.items():
            for w in words:
                if w not in self.abnormal_embeddings:
                    self.abnormal_embeddings[w] = self._get_word_embedding(w)

    def _get_word_embedding(self, word: str):
        inputs = self.tokenizer(word, return_tensors='pt', truncation=True, max_length=32).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state  # (1, seq_len, hidden)
        word_emb = embeddings.mean(dim=1).squeeze().cpu().numpy()
        return word_emb

    def _most_similar_abnormal(self, target_word, candidates):
        if not candidates:
            return None
        target_emb = self._get_word_embedding(target_word)
        similarities = []
        for w in candidates:
            cand_emb = self.abnormal_embeddings[w]
            sim = np.dot(target_emb, cand_emb) / (np.linalg.norm(target_emb) * np.linalg.norm(cand_emb) + 1e-8)
            similarities.append(sim)
        best_idx = np.argmax(similarities)
        return candidates[best_idx]

    def augment_sentence(self, sentence: str):
        words = list(jieba.cut(sentence))
        new_words = []
        i = 0
        while i < len(words):
            current = words[i]
            if current in self.patterns and i + 1 < len(words):
                next_word = words[i + 1]
                candidates = self.patterns[current]
                selected = self._most_similar_abnormal(next_word, candidates)
                if selected:
                    new_words.append(current)
                    new_words.append(selected)
                    i += 2
                    continue
            new_words.append(current)
            i += 1
        return ''.join(new_words)

def main():
    parser = argparse.ArgumentParser(description="测试词表噪声增强（基于Qwen/GPT-2模型向量相似度）")
    parser.add_argument("--model_name", type=str, required=True,
                        help="模型名称，如 'uer/gpt2-chinese-cluecorpussmall' 或 'Qwen/Qwen-1.8B'")
    parser.add_argument("--csv", required=True, help="prev_clean_summary.csv 文件路径")
    parser.add_argument("--text", type=str, help="待增强的单句")
    parser.add_argument("--sentences", nargs='+', help="多个句子，空格分隔")
    parser.add_argument("--input", type=str, help="输入文件（每行一句）")
    parser.add_argument("--output", type=str, help="输出文件（与 --input 搭配）")
    parser.add_argument("--device", type=str, default=None, help="计算设备：cpu 或 cuda（默认自动）")
    args = parser.parse_args()

    print(f"加载模型 {args.model_name} ...")
    augmenter = ModelSimilarityAugmenter(args.model_name, args.csv, args.device)
    print("模型加载完成。")

    if args.sentences:
        for i, sent in enumerate(args.sentences, 1):
            original = sent.strip()
            if not original:
                continue
            augmented = augmenter.augment_sentence(original)
            print(f"[{i}] 原始: {original}")
            print(f"    增强: {augmented}\n")
    elif args.text:
        original = args.text.strip()
        augmented = augmenter.augment_sentence(original)
        print(f"原始: {original}")
        print(f"增强: {augmented}")
    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"错误：输入文件不存在 {input_path}")
            return
        output_path = Path(args.output) if args.output else input_path.with_suffix(".model_sim_aug.txt")
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                line = line.strip()
                if line:
                    aug_line = augmenter.augment_sentence(line)
                    fout.write(aug_line + '\n')
        print(f"增强完成，结果保存至: {output_path}")
    else:
        print("请提供 --sentences 或 --text 或 --input")

if __name__ == "__main__":
    main()