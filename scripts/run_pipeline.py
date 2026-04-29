#!/usr/bin/env python3
"""
全自动数据清洗流水线
用法: python run_pipeline.py --config pipeline_config.yaml [--step split|...]
"""

import yaml
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # 替换路径中的{task_name}变量
    task_name = config['task_name']
    def replacer(s):
        if isinstance(s, str):
            return s.format(task_name=task_name, timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"))
        return s
    config = recursive_replace(config, replacer)
    return config

def run_step(step_name, cmd_args, step_config):
    """执行单个步骤，可传入配置参数"""
    print(f"\n[STEP] {step_name}")
    # 构建命令行，例如：python 01_split_dialogues.py --input ... --output ...
    cmd = [sys.executable, step_name] + cmd_args
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"步骤 {step_name} 失败，退出")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--step", choices=['all','00','01','02','03','04','05'], default='all')
    args = parser.parse_args()

    config = load_config(args.config)

    # 步骤 00：生成原始 raw_dialogues.json
    if args.step in ['all','00']:
        run_step("00_dataset_process.py", [
            "--raw_dir", config['data_source']['raw_dialogues_dir'],
            "--prompt_dir", config['data_source']['prompt_dir'],
            "--output", config['data_source']['raw_json_output']
        ], config)

    # 步骤 01：拆分为 samples，生成轮次统计
    if args.step in ['all','01']:
        run_step("01_split_dialogues.py", [
            "--input", config['processing']['split']['input_json'],
            "--output_dir", config['processing']['split']['output_samples_dir'],
            "--stats_dir", config['processing']['split']['stats_dir'],
            "--batch_size", str(config['processing']['split']['batch_size'])
        ], config)

    # 步骤 02：自动分桶（需要先读取 turn_distribution.json 并生成桶边界）
    if args.step in ['all','02']:
        # 将自动生成的桶边界写入临时文件或通过环境变量传递
        bucket_config = generate_bucket_config(config)  # 实现见下文
        run_step("02_split_into_buckets.py", [
            "--samples_dir", config['processing']['split']['output_samples_dir'],
            "--buckets_config", json.dumps(bucket_config)  # 或者写入文件
        ], config)

    # 类似处理步骤 03、04、05...

if __name__ == "__main__":
    main()