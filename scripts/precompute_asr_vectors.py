#!/usr/bin/env python3
"""
预处理ASR噪声词表（支持空格分隔的异常词列表）
用法: python scripts/precompute_asr_vectors.py --csv_path path/to/prev_clean_summary.csv --model_path /path/to/local/model
"""
import argparse
import pickle
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pypinyin import pinyin, Style

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="清洗后的词表 CSV，需包含 prev_word, abnormal_words 列")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（若不指定，则自动使用 CSV 文件所在目录）")
    parser.add_argument("--model_path", required=True, help="本地 SentenceTransformer 模型文件夹路径")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载词表
    df = pd.read_csv(args.csv_path)
    required_cols = ['prev_word', 'abnormal_words']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV文件缺少 {col} 列")

    # 1. 拆分异常词（空格分隔），收集所有独立的异常词
    all_abnormal_words = set()
    prev_to_abnormals = {}

    for _, row in df.iterrows():
        prev = str(row['prev_word']).strip()
        ab_str = str(row['abnormal_words']).strip()
        if not ab_str:
            continue
        # 按空格拆分（多个连续空格处理为单个）
        abnormal_list = [w for w in ab_str.split() if w]
        # 添加到全局集合
        all_abnormal_words.update(abnormal_list)
        # 构建前置词映射
        if prev:
            prev_to_abnormals.setdefault(prev, set()).update(abnormal_list)

    abnormal_words = sorted(list(all_abnormal_words))  # 排序保证可重现
    print(f"异常词总数（拆分后）: {len(abnormal_words)}")

    # 2. 加载本地模型，计算向量
    print(f"加载本地模型: {args.model_path}")
    model = SentenceTransformer(args.model_path)
    print("计算异常词向量...")
    vectors = model.encode(abnormal_words, show_progress_bar=True, convert_to_numpy=True)

    # 3. 保存向量
    vector_file = output_dir / "abnormal_vectors.pkl"
    with open(vector_file, 'wb') as f:
        pickle.dump({"words": abnormal_words, "vectors": vectors}, f)
    print(f"向量已保存: {vector_file}")

    # 4. 预计算拼音
    pinyin_dict = {}
    for w in abnormal_words:
        pinyin_list = pinyin(w, style=Style.NORMAL)
        pinyin_str = ''.join([item[0] for item in pinyin_list])
        pinyin_dict[w] = pinyin_str

    pinyin_file = output_dir / "abnormal_pinyin.pkl"
    with open(pinyin_file, 'wb') as f:
        pickle.dump(pinyin_dict, f)
    print(f"拼音映射已保存: {pinyin_file}")

    # 5. 保存前置词->异常词列表映射（每个前置词对应的异常词已去重）
    prev_to_abnormals = {k: list(v) for k, v in prev_to_abnormals.items()}
    prev_file = output_dir / "prev_to_abnormals.pkl"
    with open(prev_file, 'wb') as f:
        pickle.dump(prev_to_abnormals, f)
    print(f"前置词映射已保存: {prev_file}")

    print("预处理完成！")
    print(f"统计: 前置词数量 {len(prev_to_abnormals)}，异常词数量 {len(abnormal_words)}")

if __name__ == "__main__":
    main()