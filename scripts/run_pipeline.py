#!/usr/bin/env python3
"""
全自动数据清洗与增强流水线
用法: python scripts/run_pipeline.py --config pipeline_config.yaml [--step STEP_NAME]
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

# 全局日志对象（会在 setup_logging 中初始化）
logger = None

def setup_logging(task_name: str, log_dir: Path, console_level=logging.INFO, file_level=logging.DEBUG):
    """配置日志：控制台 INFO，文件 DEBUG，输出到任务隔离目录下的 logs/"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pipeline_{task_name}.log"
    
    log = logging.getLogger("Pipeline")
    log.setLevel(logging.DEBUG)
    if log.handlers:
        log.handlers.clear()
    
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(file_level)
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    log.addHandler(fh)
    log.addHandler(ch)
    return log

def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件，并替换其中的 {task_name}、{timestamp} 等变量"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    task_name = config['task_name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def recursive_replace(obj):
        if isinstance(obj, dict):
            return {k: recursive_replace(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_replace(item) for item in obj]
        elif isinstance(obj, str):
            return obj.format(task_name=task_name, timestamp=timestamp)
        else:
            return obj
    
    return recursive_replace(config)

def get_step_done_flag(task_dir: Path, step_name: str) -> Path:
    """返回步骤完成的标记文件路径"""
    return task_dir / f".step_{step_name}_done"

def is_step_completed(task_dir: Path, step_name: str) -> bool:
    """检查步骤是否已完成（断点续跑）"""
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
    
    # 构建命令行参数：将全局配置以 JSON 形式传递给子脚本
    cmd = [sys.executable, script_path, '--config_json', json.dumps(global_config)]
    
    # 针对特定步骤添加额外参数（也可完全依赖 config_json，但为保持灵活性保留）
    if step_key == '01_split':
        batch_size = step_config.get('batch_size', 120000)
        cmd.extend(['--batch_size', str(batch_size)])
    elif step_key == '02_bucket':
        strategy = step_config.get('strategy', 'auto')
        cmd.extend(['--strategy', strategy])
        if step_config.get('auto_params'):
            cmd.extend(['--auto_params', json.dumps(step_config['auto_params'])])
        if step_config.get('manual_buckets'):
            cmd.extend(['--manual_buckets', json.dumps(step_config['manual_buckets'])])
    elif step_key == '03_clean':
        configs_dir = step_config.get('configs_dir', 'configs/configs_qa')
        bucket_config_map = step_config.get('bucket_config_map', [])
        tag = step_config.get('tag', task_name)
        cmd.extend(['--configs_dir', configs_dir])
        cmd.extend(['--bucket_config_map', json.dumps(bucket_config_map)])
        cmd.extend(['--tag', tag])
    elif step_key == '05_augment':
        num_variants = step_config.get('num_variants', 3)
        target_roles = step_config.get('target_roles', ['user'])
        cmd.extend(['--num_variants', str(num_variants)])
        cmd.extend(['--target_roles'] + target_roles)
        if step_config.get('only_loss_true'):
            cmd.append('--only_loss_true')
        if step_config.get('adaptive_variants'):
            cmd.append('--adaptive_variants')
        cmd.extend(['--tag', step_config.get('tag', task_name)])
    
    # 执行子进程
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"步骤 {step_key} 执行失败，返回码 {result.returncode}")
            logger.error(f"STDERR: {result.stderr[:1000]}")
            # 可选：将完整输出写入日志文件
            error_log = task_dir / "logs" / f"{step_key}_error.log"
            error_log.parent.mkdir(exist_ok=True)
            with open(error_log, 'w') as f:
                f.write(f"Return code: {result.returncode}\n")
                f.write("STDOUT:\n" + result.stdout)
                f.write("\nSTDERR:\n" + result.stderr)
            logger.info(f"完整错误日志已保存到 {error_log}")
            return False
        logger.info(f"步骤 {step_key} 执行成功")
        mark_step_done(task_dir, step_key)
        return True
    except Exception as e:
        logger.exception(f"步骤 {step_key} 执行异常: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="数据清洗与增强流水线")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    parser.add_argument("--step", help="单独运行某个步骤（如 00_generate_raw）", default=None)
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    task_name = config['task_name']
    base_dir = Path(config['paths']['output']['base_dir'])
    task_dir = base_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    global logger
    log_dir = task_dir / "logs"
    console_level = getattr(logging, config['logging'].get('level', 'INFO'))
    file_level = getattr(logging, config['logging'].get('file_level', 'DEBUG'))
    logger = setup_logging(task_name, log_dir, console_level, file_level)
    
    logger.info(f"任务启动: {task_name}")
    logger.info(f"任务目录: {task_dir}")
    logger.info(f"断点续跑模式: {config.get('resume', False)}")
    
    # 定义步骤执行顺序（必须与配置文件中的 steps 键名一致）
    steps_order = ['00_generate_raw', '01_split', '02_bucket', '03_clean', '04_finalize', '05_augment']
    
    # 如果指定了单步，只执行该步骤
    if args.step:
        if args.step not in config['steps']:
            logger.error(f"配置文件中未找到步骤定义: {args.step}")
            sys.exit(1)
        step_config = config['steps'][args.step]
        success = run_step(step_config, args.step, task_dir, task_name, config)
        sys.exit(0 if success else 1)
    
    # 顺序执行所有步骤
    for step_key in steps_order:
        if step_key not in config['steps']:
            logger.warning(f"配置文件中缺少步骤定义: {step_key}，跳过")
            continue
        step_config = config['steps'][step_key]
        if not step_config.get('enabled', True):
            logger.info(f"步骤 {step_key} 已禁用，跳过")
            continue
        success = run_step(step_config, step_key, task_dir, task_name, config)
        if not success:
            logger.error(f"流水线在步骤 {step_key} 中断")
            sys.exit(1)
    
    logger.info("所有步骤执行完毕！")

if __name__ == "__main__":
    main()