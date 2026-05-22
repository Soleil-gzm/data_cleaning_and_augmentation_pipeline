#!/usr/bin/env python3
"""
预处理ASR噪声词表（支持空格分隔的异常词列表）
用法: 
  命令行模式: python scripts/precompute_asr_vectors.py --csv_path <path> --model_path <path> [--output_dir <path>]
  硬编码模式: python scripts/precompute_asr_vectors.py --hardcoded
"""
import argparse
import pickle
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pypinyin import pinyin, Style

# ========== 硬编码模式配置（请修改为实际路径）==========
HARDCODED_CSV_PATH = "resources/prev_clean/sample_20/qwen/prev_clean_prev_window_1/prev_clean_summary.csv"
HARDCODED_OUTPUT_DIR = None  # 设为 None 表示使用 CSV 所在目录，或指定具体路径
HARDCODED_MODEL_PATH = "/home/zimeng/projects/data_process/models/paraphrase-multilingual-MiniLM-L12-v2"
# ==================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardcoded", action="store_true", help="使用硬编码路径（需编辑脚本内部变量）")
    parser.add_argument("--csv_path", type=str, help="清洗后的词表 CSV")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--model_path", type=str, help="本地 SentenceTransformer 模型文件夹路径")
    args = parser.parse_args()

    if args.hardcoded:
        csv_path = Path(HARDCODED_CSV_PATH)
        model_path = HARDCODED_MODEL_PATH
        if HARDCODED_OUTPUT_DIR:
            output_dir = Path(HARDCODED_OUTPUT_DIR)
        else:
            output_dir = csv_path.parent
        print(f"使用硬编码模式:")
        print(f"  CSV路径: {csv_path}")
        print(f"  模型路径: {model_path}")
        print(f"  输出目录: {output_dir}")
    else:
        if not args.csv_path or not args.model_path:
            parser.error("非硬编码模式下必须提供 --csv_path 和 --model_path")
        csv_path = Path(args.csv_path)
        model_path = args.model_path
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = csv_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载词表
    df = pd.read_csv(csv_path)
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
        abnormal_list = [w for w in ab_str.split() if w]
        all_abnormal_words.update(abnormal_list)
        if prev:
            prev_to_abnormals.setdefault(prev, set()).update(abnormal_list)

    abnormal_words = sorted(list(all_abnormal_words))
    print(f"异常词总数（拆分后）: {len(abnormal_words)}")

    # 2. 加载本地模型，计算向量
    print(f"加载本地模型: {model_path}")
    model = SentenceTransformer(model_path)
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

    # 5. 保存前置词->异常词列表映射
    prev_to_abnormals = {k: list(v) for k, v in prev_to_abnormals.items()}
    prev_file = output_dir / "prev_to_abnormals.pkl"
    with open(prev_file, 'wb') as f:
        pickle.dump(prev_to_abnormals, f)
    print(f"前置词映射已保存: {prev_file}")

    print("预处理完成！")
    print(f"统计: 前置词数量 {len(prev_to_abnormals)}，异常词数量 {len(abnormal_words)}")

if __name__ == "__main__":
    main()