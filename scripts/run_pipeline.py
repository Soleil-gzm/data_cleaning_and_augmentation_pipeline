#!/usr/bin/env python3
"""
全自动流水线主控脚本
支持断点续跑、统一日志、配置文件驱动
用法: python run_pipeline.py --config pipeline_config.yaml [--step STEP_NAME]
"""

import os
import sys
import yaml
import json
import logging
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 全局日志对象
logger = None

def setup_logging(task_name: str, log_dir: Path, console_level=logging.INFO, file_level=logging.DEBUG):
    """配置日志：控制台+文件，支持任务隔离"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{task_name}.log"
    
    log = logging.getLogger("Pipeline")
    log.setLevel(logging.DEBUG)
    # 避免重复添加handler
    if log.handlers:
        log.handlers.clear()
    
    # 文件handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(file_level)
    # 控制台handler
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    log.addHandler(fh)
    log.addHandler(ch)
    return log

def load_config(config_path: str) -> Dict[str, Any]:
    """加载并预处理配置，替换路径中的{task_name}等变量"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    task_name = config['task_name']
    # 递归替换字符串中的变量
    def replace_vars(obj):
        if isinstance(obj, dict):
            return {k: replace_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_vars(item) for item in obj]
        elif isinstance(obj, str):
            return obj.format(task_name=task_name, timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"))
        else:
            return obj
    
    return replace_vars(config)

def get_step_done_flag(task_dir: Path, step_name: str) -> Path:
    """返回某步骤完成的标记文件路径"""
    return task_dir / f".step_{step_name}_done"

def is_step_completed(task_dir: Path, step_name: str) -> bool:
    """检查步骤是否已完成"""
    return get_step_done_flag(task_dir, step_name).exists()

def mark_step_done(task_dir: Path, step_name: str):
    """标记步骤完成"""
    flag = get_step_done_flag(task_dir, step_name)
    flag.touch()

def run_step(step_config: dict, step_key: str, task_dir: Path, task_name: str, global_config: dict):
    """执行单个步骤，支持断点续跑"""
    if not step_config.get('enabled', True):
        logger.info(f"步骤 {step_key} 已禁用，跳过")
        return True
    
    if global_config.get('resume', False) and is_step_completed(task_dir, step_key):
        logger.info(f"步骤 {step_key} 已完成，跳过（断点续跑）")
        return True
    
    logger.info(f"开始执行步骤: {step_key}")
    script_path = step_config.get('script')
    if not script_path or not Path(script_path).exists():
        logger.error(f"脚本不存在: {script_path}")
        return False
    
    # 构建命令行参数，将配置以JSON形式传递给脚本
    # 同时传递任务隔离目录等公共参数
    cmd = [sys.executable, script_path, '--config_json', json.dumps(global_config)]
    # 针对不同步骤可添加额外参数
    if step_key == '01_split':
        cmd.extend(['--batch_size', str(step_config.get('batch_size', 120000))])
    elif step_key == '02_bucket':
        cmd.extend(['--strategy', step_config.get('strategy', 'auto')])
        if step_config.get('auto_params'):
            cmd.extend(['--auto_params', json.dumps(step_config['auto_params'])])
        if step_config.get('manual_buckets'):
            cmd.extend(['--manual_buckets', json.dumps(step_config['manual_buckets'])])
    elif step_key == '03_clean':
        cmd.extend(['--configs_dir', step_config.get('configs_dir', 'configs/configs_qa')])
        cmd.extend(['--bucket_config_map', json.dumps(step_config.get('bucket_config_map', []))])
    elif step_key == '05_augment':
        cmd.extend(['--num_variants', str(step_config.get('num_variants', 3))])
        cmd.extend(['--target_roles'] + step_config.get('target_roles', ['user']))
    
    # 执行
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"步骤 {step_key} 执行失败，返回码 {result.returncode}")
            logger.error(f"STDERR: {result.stderr[:1000]}")
            return False
        logger.info(f"步骤 {step_key} 执行成功")
        mark_step_done(task_dir, step_key)
        return True
    except Exception as e:
        logger.exception(f"步骤 {step_key} 执行异常: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML配置文件路径")
    parser.add_argument("--step", help="单独运行某个步骤 (如 00_generate_raw)", default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    task_name = config['task_name']
    base_dir = Path(config['paths']['output']['base_dir'])
    task_dir = base_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置全局日志
    global logger
    log_dir = task_dir / "logs"
    console_level = getattr(logging, config['logging'].get('level', 'INFO'))
    file_level = getattr(logging, config['logging'].get('file_level', 'DEBUG'))
    logger = setup_logging(task_name, log_dir, console_level, file_level)
    
    logger.info(f"任务启动: {task_name}, 任务目录: {task_dir}")
    logger.info(f"断点续跑模式: {config.get('resume', False)}")
    
    steps_order = ['00_generate_raw', '01_split', '02_bucket', '03_clean', '04_finalize', '05_augment']
    
    if args.step:
        if args.step in config['steps']:
            step_config = config['steps'][args.step]
            success = run_step(step_config, args.step, task_dir, task_name, config)
            sys.exit(0 if success else 1)
        else:
            logger.error(f"未找到步骤: {args.step}")
            sys.exit(1)
    
    # 顺序执行所有步骤
    for step_key in steps_order:
        if step_key not in config['steps']:
            logger.warning(f"配置中缺少步骤定义: {step_key}，跳过")
            continue
        step_config = config['steps'][step_key]
        if not step_config.get('enabled', True):
            continue
        success = run_step(step_config, step_key, task_dir, task_name, config)
        if not success:
            logger.error(f"流水线在步骤 {step_key} 中断")
            sys.exit(1)
    
    logger.info("所有步骤执行完毕！")

if __name__ == "__main__":
    main()