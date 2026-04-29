#!/usr/bin/env python3
"""
split_into_buckets.py
将 samples/ 目录下的所有 JSONL 文件按 turn 值分桶，输出到 bucketed/ 目录
桶定义：
   bucket_0_2  : turn 0,1,2
   bucket_3_5  : turn 3,4,5
   bucket_6_10 : turn 6,7,8,9,10
   bucket_11_20: turn 11..20
   bucket_21plus: turn >=21
"""

import json
import os
from pathlib import Path
from collections import defaultdict

# 配置
INPUT_DIR = "task_20260429/task_92049/output_cleaning/samples"
OUTPUT_BASE = "task_20260429/task_92049/output_cleaning/bucketed"

BUCKETS = {
    (0, 0): "bucket_0",
    (1, 1):"bucket_1",
    (2, 2): "bucket_2",
    (3, 3): "bucket_3",
    (4, 4): "bucket_4",
    (5, 5): "bucket_5",
    (6, 6): "bucket_6",
    (7, 7): "bucket_7",
    (8, 8): "bucket_8",
    (9, 9): "bucket_9",
    (10, float('inf')): "bucket_10plus",
}

# task_20260429/task_92049/
BUCKETS = {
    (0, 0): "bucket_0",
    (1, 1):"bucket_1",
    (2, 2): "bucket_2",
    (3, 3): "bucket_3",
    (4, 4): "bucket_4",
    (5, 5): "bucket_5",
    (6, 6): "bucket_6",
    (7, 7): "bucket_7",
    (8, 8): "bucket_8",
    (9, 9): "bucket_9",
    (10, 10): "bucket_10",
    (11, 11): "bucket_11",
    (12, 12): "bucket_12",
    (13, 22): "bucket_13_22",
    (23, float('inf')): "bucket_23plus",
}

'''
根据 turn 值查找对应的桶名称。
遍历 BUCKETS 的每一项，如果 low <= turn <= high 则返回该桶名。
如果没找到（理论上不会发生，因为最后一个桶覆盖到无穷大），则返回 "bucket_23plus" 作为备用名称。
'''
def load_turn_distribution(stats_dir):
    """从 stats_dir/turn_distribution.json 加载轮次统计"""
    stats_file = Path(stats_dir) / "turn_distribution.json"
    if not stats_file.exists():
        raise FileNotFoundError(f"未找到轮次统计文件: {stats_file}")
    with open(stats_file, 'r') as f:
        data = json.load(f)
    return data['turn_distribution']   # { turn: count }

def auto_buckets_from_distribution(turn_dist, strategy="percentile", params=None):
    """
    根据轮次分布自动生成桶边界列表。
    返回: list of (low, high) 区间，如 [(0,0), (1,2), (3,5), ...]
    """
    turns = sorted([int(k) for k in turn_dist.keys()])
    counts = [turn_dist[t] for t in turns]
    cumulative = []
    total = sum(counts)
    running = 0
    for cnt in counts:
        running += cnt
        cumulative.append(running / total)

    if strategy == "percentile":
        percentiles = params.get('percentiles', [0, 25, 50, 75, 90, 95, 100])
        # 找到每个百分位对应的 turn
        boundaries = set()
        for p in percentiles:
            target = p / 100.0
            # 找到第一个累计比例 >= target 的索引
            for i, cum in enumerate(cumulative):
                if cum >= target:
                    boundaries.add(turns[i])
                    break
        boundaries = sorted(boundaries)
        # 生成区间
        buckets = []
        for i in range(len(boundaries)-1):
            low = boundaries[i]
            high = boundaries[i+1] - 1 if boundaries[i+1] > boundaries[i] else boundaries[i]
            if low <= high:
                buckets.append((low, high))
        # 特殊处理最后一个桶到无穷
        last_turn = boundaries[-1]
        buckets.append((last_turn, float('inf')))
        return buckets

    elif strategy == "equal_count":
        min_bucket_size = params.get('min_bucket_size', 1000)
        buckets = []
        start = turns[0]
        cum_cnt = 0
        for t, cnt in zip(turns, counts):
            cum_cnt += cnt
            if cum_cnt >= min_bucket_size:
                buckets.append((start, t))
                start = t + 1
                cum_cnt = 0
        if start <= turns[-1]:
            buckets.append((start, float('inf')))
        return buckets

    else:
        raise ValueError(f"未知策略: {strategy}")

def get_bucket_name(turn):
    for (low, high), name in BUCKETS.items():
        if low <= turn <= high:
            return name
    return "bucket_23plus"  # fallback

def main():
    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        print(f"错误：目录 {INPUT_DIR} 不存在，请先运行拆分脚本生成 samples/")
        return

    output_base = Path(OUTPUT_BASE)
    output_base.mkdir(parents=True, exist_ok=True)

    # 为每个桶创建子目录并清空（避免旧数据干扰）
    bucket_dirs = {}
    for name in set(get_bucket_name(i) for i in range(0, 100)):
        bucket_dir = output_base / name
        bucket_dir.mkdir(exist_ok=True)
        # 清空目录内容（可选，注释掉则不清空）
        for f in bucket_dir.glob("*.jsonl"):
            f.unlink()
        bucket_dirs[name] = bucket_dir

    # 遍历所有 JSONL 文件
    jsonl_files = list(input_path.glob("*.jsonl"))
    print(f"找到 {len(jsonl_files)} 个 JSONL 文件")

    # 为每个桶准备写入文件（保持原文件名，但放入对应桶目录）
    # 由于一个文件内可能包含多种 turn，我们需要为每个桶动态打开文件
    # 简单起见：对每个输入文件，遍历其行，按桶写入多个输出文件（同名）
    for input_file in jsonl_files:
        print(f"处理: {input_file.name}")
        # 为每个桶准备该文件的输出句柄（延迟打开）
        file_handles = {}
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    turn = data.get('turn')
                    if turn is None:
                        continue
                    bucket = get_bucket_name(turn)
                    # 获取输出文件路径
                    output_file = bucket_dirs[bucket] / input_file.name
                    # 打开文件句柄（追加模式）
                    if output_file not in file_handles:
                        file_handles[output_file] = open(output_file, 'a', encoding='utf-8')
                    file_handles[output_file].write(line)
        finally:
            for h in file_handles.values():
                h.close()

    # 打印统计
    print("\n分桶完成，各桶文件统计：")
    for name, dir_path in bucket_dirs.items():
        count = sum(1 for _ in dir_path.glob("*.jsonl"))
        print(f"  {name}: {count} 个文件")

if __name__ == "__main__":
    main()