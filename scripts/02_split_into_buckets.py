#!/usr/bin/env python3
"""
分桶脚本（增强版）
支持手动/自动分桶策略，自动读取 turn_distribution.json 生成桶边界
用法: python 02_split_into_buckets.py --config_json '{"task_name":"xxx", ...}'
     或直接传递 --samples_dir --buckets_json
"""

import json
import os
import sys
import argparse
import logging
from pathlib import Path
from collections import defaultdict

def setup_logger(task_dir=None):
    logger = logging.getLogger("BucketSplit")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = setup_logger()

def load_turn_distribution(stats_dir):
    """加载轮次分布文件"""
    stats_file = Path(stats_dir) / "turn_distribution.json"
    if not stats_file.exists():
        raise FileNotFoundError(f"未找到轮次统计文件: {stats_file}")
    with open(stats_file, 'r') as f:
        data = json.load(f)
    return {int(k): v for k, v in data.get('turn_distribution', {}).items()}

def auto_buckets_from_distribution(turn_dist, strategy, params=None):
    """
    自动生成桶边界
    :param turn_dist: dict {turn: count}
    :param strategy: "percentile" 或 "equal_count"
    :param params: 对应的参数字典
    """
    turns = sorted(turn_dist.keys())
    counts = [turn_dist[t] for t in turns]
    cumulative = []
    total = sum(counts)
    running = 0
    for cnt in counts:
        running += cnt
        cumulative.append(running / total)
    
    if strategy == "percentile":
        percentiles = params.get('percentiles', [0, 25, 50, 75, 90, 95, 100])
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
        # 最后一个桶到无穷
        last = boundaries[-1]
        buckets.append((last, float('inf')))
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

def get_bucket_name(turn, buckets):
    """根据桶边界列表获取桶名"""
    for idx, (low, high) in enumerate(buckets):
        if low <= turn <= high:
            return f"bucket_{low}_{high if high != float('inf') else 'plus'}"
    return "bucket_unknown"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, help="全局配置JSON字符串")
    parser.add_argument("--samples_dir", type=str, help="samples目录路径")
    parser.add_argument("--output_base", type=str, help="输出桶根目录")
    parser.add_argument("--strategy", type=str, choices=['auto','manual','percentile','equal_count'], default='auto')
    parser.add_argument("--auto_params", type=str, help="自动分桶参数JSON")
    parser.add_argument("--manual_buckets", type=str, help="手动桶边界JSON")
    args = parser.parse_args()
    
    # 解析配置
    if args.config_json:
        config = json.loads(args.config_json)
        task_name = config['task_name']
        base_dir = Path(config['paths']['output']['base_dir'])
        samples_dir = base_dir / task_name / "samples"
        output_base = base_dir / task_name / "bucketed"
        # 从steps.02_bucket中获取策略
        bucket_cfg = config.get('steps', {}).get('02_bucket', {})
        strategy = bucket_cfg.get('strategy', 'auto')
        auto_params = bucket_cfg.get('auto_params', {})
        manual_buckets = bucket_cfg.get('manual_buckets', [])
    else:
        # 直接命令行模式
        samples_dir = Path(args.samples_dir)
        output_base = Path(args.output_base) if args.output_base else Path("bucketed")
        strategy = args.strategy
        auto_params = json.loads(args.auto_params) if args.auto_params else {}
        manual_buckets = json.loads(args.manual_buckets) if args.manual_buckets else []
    
    logger.info(f"分桶输入目录: {samples_dir}")
    logger.info(f"输出根目录: {output_base}")
    
    # 获取轮次分布
    stats_dir = samples_dir.parent / "stats"   # 假设stats在samples同级目录
    turn_dist = load_turn_distribution(stats_dir)
    logger.info(f"加载轮次分布，共 {len(turn_dist)} 种轮次")
    
    # 生成桶边界
    if strategy == 'auto' or strategy == 'percentile' or strategy == 'equal_count':
        # 自动策略统一处理
        if strategy == 'auto':
            strategy = 'percentile'   # 默认百分位
        buckets = auto_buckets_from_distribution(turn_dist, strategy, auto_params)
        logger.info(f"自动生成 {len(buckets)} 个桶: {buckets}")
    elif strategy == 'manual':
        if not manual_buckets:
            raise ValueError("手动策略但未提供 manual_buckets")
        buckets = [(low, high) for low, high in manual_buckets]
    else:
        raise ValueError(f"不支持的分桶策略: {strategy}")
    
    # 建立桶名到输出目录的映射
    output_base.mkdir(parents=True, exist_ok=True)
    bucket_dirs = {}
    for idx, (low, high) in enumerate(buckets):
        if high == float('inf'):
            bucket_name = f"bucket_{low}_plus"
        else:
            bucket_name = f"bucket_{low}_{high}"
        bucket_dir = output_base / bucket_name
        bucket_dir.mkdir(exist_ok=True)
        bucket_dirs[bucket_name] = bucket_dir
        # 清空目录内容（可选）
        for f in bucket_dir.glob("*.jsonl"):
            f.unlink()
    
    # 遍历所有jsonl文件并分桶
    jsonl_files = list(samples_dir.glob("*.jsonl"))
    logger.info(f"找到 {len(jsonl_files)} 个样本文件")
    for input_file in jsonl_files:
        logger.debug(f"处理文件: {input_file.name}")
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
                    bucket_name = get_bucket_name(turn, buckets)
                    bucket_dir = bucket_dirs.get(bucket_name)
                    if not bucket_dir:
                        logger.warning(f"turn {turn} 未匹配到桶，跳过")
                        continue
                    out_file = bucket_dir / input_file.name
                    if out_file not in file_handles:
                        file_handles[out_file] = open(out_file, 'a', encoding='utf-8')
                    file_handles[out_file].write(line)
        finally:
            for h in file_handles.values():
                h.close()
    
    logger.info(f"分桶完成，输出目录: {output_base}")
    for name, d in bucket_dirs.items():
        cnt = sum(1 for _ in d.glob("*.jsonl"))
        logger.info(f"  {name}: {cnt} 个文件")

if __name__ == "__main__":
    main()