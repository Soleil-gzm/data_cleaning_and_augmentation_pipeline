#!/usr/bin/env python3
"""
分桶脚本（配置驱动版，支持自动清理旧桶）
从 samples 目录读取 JSONL 文件，按 turn 值分桶输出。
支持自动分桶（基于轮次分布百分位或等频）和手动分桶。
每次运行前自动清空输出目录，避免数据累积。
"""

import json
import os
import sys
import argparse
import logging
import shutil
from pathlib import Path
from collections import defaultdict

# ========== 日志配置 ==========
def setup_logger(task_dir, task_name):
    log_dir = task_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"02_split_bucket_{task_name}.log"

    logger = logging.getLogger("BucketSplit")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def load_turn_distribution(stats_dir):
    stats_file = Path(stats_dir) / "turn_distribution.json"
    if not stats_file.exists():
        raise FileNotFoundError(f"未找到轮次统计文件: {stats_file}")
    with open(stats_file, 'r') as f:
        data = json.load(f)
    return {int(k): v for k, v in data.get('turn_distribution', {}).items()}

def auto_buckets_from_distribution(turn_dist, strategy, params=None):
    turns = sorted(turn_dist.keys())
    counts = [turn_dist[t] for t in turns]
    total = sum(counts)
    
    if strategy == "percentile":
        percentiles = params.get('percentiles', [0, 25, 50, 75, 90, 95, 100])
        cumulative = []
        running = 0
        for cnt in counts:
            running += cnt
            cumulative.append(running / total)
        boundaries = set()
        for p in percentiles:
            target = p / 100.0
            for i, cum in enumerate(cumulative):
                if cum >= target:
                    boundaries.add(turns[i])
                    break
        boundaries = sorted(boundaries)
        buckets = []
        for i in range(len(boundaries)-1):
            low = boundaries[i]
            high = boundaries[i+1] - 1 if boundaries[i+1] > boundaries[i] else boundaries[i]
            if low <= high:
                buckets.append((low, high))
        buckets.append((boundaries[-1], float('inf')))
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
        raise ValueError(f"未知自动分桶策略: {strategy}")

def get_bucket_name(bucket_idx, low, high):
    if high == float('inf'):
        return f"bucket_{low}_plus"
    elif low == high:
        return f"bucket_{low}"
    else:
        return f"bucket_{low}_{high}"

def get_bucket_for_turn(turn, buckets):
    for idx, (low, high) in enumerate(buckets):
        if low <= turn <= high:
            return idx, get_bucket_name(idx, low, high)
    return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, help="全局配置JSON字符串")
    parser.add_argument("--samples_dir", type=str, help="samples目录路径")
    parser.add_argument("--output_base", type=str, help="输出桶根目录")
    parser.add_argument("--strategy", type=str, choices=['auto', 'manual', 'percentile', 'equal_count'], default='auto')
    parser.add_argument("--auto_params", type=str, help="自动分桶参数JSON")
    parser.add_argument("--manual_buckets", type=str, help="手动桶边界JSON")
    args = parser.parse_args()

    if args.config_json:
        config = json.loads(args.config_json)
        task_name = config['task_name']
        base_dir = Path(config['paths']['output']['base_dir'])
        task_dir = base_dir / task_name
        step_cfg = config.get('steps', {}).get('02_bucket', {})
        samples_dir = step_cfg.get('samples_dir') or (task_dir / "samples")
        output_base = step_cfg.get('output_base') or (task_dir / "bucketed")
        strategy = step_cfg.get('strategy', 'auto')
        auto_params = step_cfg.get('auto_params', {})
        manual_buckets = step_cfg.get('manual_buckets', [])
        logger = setup_logger(task_dir, task_name)
        logger.info(f"任务名称: {task_name}")
        logger.info(f"任务目录: {task_dir}")
    else:
        samples_dir = args.samples_dir
        output_base = args.output_base
        strategy = args.strategy
        auto_params = json.loads(args.auto_params) if args.auto_params else {}
        manual_buckets = json.loads(args.manual_buckets) if args.manual_buckets else []
        if not samples_dir or not output_base:
            print("错误：独立模式需要提供 --samples_dir 和 --output_base")
            sys.exit(1)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("BucketSplit")
        task_dir = Path(samples_dir).parent

    samples_dir = Path(samples_dir)
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    # ========== 新增：清空旧的桶内容 ==========
    logger.info(f"清空旧的分桶目录: {output_base}")
    for item in output_base.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except Exception as e:
            logger.warning(f"删除 {item} 失败: {e}")
    # ========================================

    logger.info(f"样本目录: {samples_dir}")
    logger.info(f"输出根目录: {output_base}")

    # 加载轮次分布（自动模式需要）
    if strategy in ['auto', 'percentile', 'equal_count']:
        stats_dir = task_dir / "stats" if 'task_dir' in locals() else samples_dir.parent / "stats"
        if not stats_dir.exists():
            logger.error(f"统计目录不存在: {stats_dir}，请先运行 01_split_dialogues.py")
            sys.exit(1)
        turn_dist = load_turn_distribution(stats_dir)
        logger.info(f"加载轮次分布，共 {len(turn_dist)} 种轮次，总样本数: {sum(turn_dist.values())}")
    else:
        turn_dist = None

    # 生成桶边界
    if strategy == 'auto':
        actual_strategy = 'percentile'
        params = auto_params if auto_params else {'percentiles': [0, 25, 50, 75, 90, 95, 100]}
        buckets = auto_buckets_from_distribution(turn_dist, actual_strategy, params)
        logger.info(f"自动分桶 (percentile) 生成 {len(buckets)} 个桶: {buckets}")
    elif strategy == 'percentile' or strategy == 'equal_count':
        buckets = auto_buckets_from_distribution(turn_dist, strategy, auto_params)
        logger.info(f"自动分桶 ({strategy}) 生成 {len(buckets)} 个桶: {buckets}")
    elif strategy == 'manual':
        if not manual_buckets:
            logger.error("手动策略但未提供 manual_buckets")
            sys.exit(1)
        buckets = [(low, high) for low, high in manual_buckets]
        logger.info(f"手动分桶，共 {len(buckets)} 个桶: {buckets}")
    else:
        logger.error(f"不支持的分桶策略: {strategy}")
        sys.exit(1)

    # 创建桶目录（已经清空过，直接新建）
    bucket_dirs = {}
    for idx, (low, high) in enumerate(buckets):
        bucket_name = get_bucket_name(idx, low, high)
        bucket_dir = output_base / bucket_name
        bucket_dir.mkdir(exist_ok=True)
        bucket_dirs[bucket_name] = bucket_dir
    logger.info(f"创建了 {len(bucket_dirs)} 个桶目录")

    # 遍历所有 JSONL 文件并分桶
    jsonl_files = list(samples_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning(f"样本目录中未找到 JSONL 文件: {samples_dir}")
        sys.exit(0)
    logger.info(f"找到 {len(jsonl_files)} 个样本文件")

    total_samples = 0
    for input_file in jsonl_files:
        logger.debug(f"处理文件: {input_file.name}")
        file_handles = {}
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    turn = data.get('turn')
                    if turn is None:
                        continue
                    _, bucket_name = get_bucket_for_turn(turn, buckets)
                    if bucket_name is None:
                        logger.warning(f"turn {turn} 未匹配到任何桶，跳过")
                        continue
                    bucket_dir = bucket_dirs[bucket_name]
                    out_file = bucket_dir / input_file.name
                    if out_file not in file_handles:
                        file_handles[out_file] = open(out_file, 'a', encoding='utf-8')
                    file_handles[out_file].write(line + '\n')
                    total_samples += 1
        finally:
            for h in file_handles.values():
                h.close()

    logger.info(f"分桶完成，共处理 {total_samples} 条样本")
    for name, d in bucket_dirs.items():
        cnt = sum(1 for _ in d.glob("*.jsonl"))
        logger.info(f"  {name}: {cnt} 个文件")

if __name__ == "__main__":
    main()