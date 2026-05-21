#!/usr/bin/env python3
"""
预处理ASR噪声词表，预计算异常词的向量和拼音串，供后续增强使用。
用法: python scripts/precompute_asr_vectors.py --csv_path path/to/prev_clean_summary.csv --output_dir ./intermediate/asr_cache
"""
import argparse
import pickle
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pypinyin import pinyin, Style
import Levenshtein  # 仅用于演示，实际拼音相似度函数可写在增强器中

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="清洗后的词表 CSV，需包含 abnormal_words 列")
    parser.add_argument("--output_dir", default="./intermediate/asr_cache", help="输出目录")
    parser.add_argument("--model_name", default="Models/paraphrase-multilingual-MiniLM-L12-v2", help="向量模型")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载词表
    df = pd.read_csv(args.csv_path)
    if 'abnormal_words' not in df.columns:
        raise ValueError("CSV文件缺少 abnormal_words 列")
    abnormal_words = df['abnormal_words'].unique().tolist()
    print(f"异常词总数: {len(abnormal_words)}")

    # 2. 预计算向量
    print(f"加载模型: {args.model_name}")
    model = SentenceTransformer(args.model_name)
    print("计算异常词向量...")
    vectors = model.encode(abnormal_words, show_progress_bar=True, convert_to_numpy=True)

    # 保存向量
    vector_file = output_dir / "abnormal_vectors.pkl"
    with open(vector_file, 'wb') as f:
        pickle.dump({"words": abnormal_words, "vectors": vectors}, f)
    print(f"向量已保存: {vector_file}")

    # 3. 预计算拼音串
    pinyin_dict = {}
    for w in abnormal_words:
        # 获取不带声调的拼音字符串（例如 "huan kuan" -> "huankuan"）
        pinyin_list = pinyin(w, style=Style.NORMAL)
        pinyin_str = ''.join([item[0] for item in pinyin_list])
        pinyin_dict[w] = pinyin_str

    pinyin_file = output_dir / "abnormal_pinyin.pkl"
    with open(pinyin_file, 'wb') as f:
        pickle.dump(pinyin_dict, f)
    print(f"拼音映射已保存: {pinyin_file}")

    # 4. 可选：构建前置词->异常词列表映射（用于候选集限制）
    if 'prev_word' in df.columns:
        prev_to_abnormals = {}
        for _, row in df.iterrows():
            prev = row['prev_word']
            ab = row['abnormal_words']
            prev_to_abnormals.setdefault(prev, set()).add(ab)
        # 转为列表
        prev_to_abnormals = {k: list(v) for k, v in prev_to_abnormals.items()}
        prev_file = output_dir / "prev_to_abnormals.pkl"
        with open(prev_file, 'wb') as f:
            pickle.dump(prev_to_abnormals, f)
        print(f"前置词映射已保存: {prev_file}")

    print("预处理完成！")

if __name__ == "__main__":
    main()