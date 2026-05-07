#!/usr/bin/env python3
"""
应用清洗结果，生成最终训练 JSON（配置驱动版）
根据 cleaned_jsonl 中保留的 (id, turn) 对，将原始 JSON 中对应 assistant 的 loss 设为 True，其余设为 False。
支持自动获取最新清洗结果或手动指定 run_id。
输出目录：final_training_data/{source_run_id}_final/cleaned_training_data.json
"""

import json
import os
import sys
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ========== 日志配置 ==========
def setup_logger(task_dir, run_id):     # 用于区分同任务下的不同清洗尝试。
    log_dir = task_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"04_write_loss_{run_id}.log"
    
    logger = logging.getLogger("Finalize")
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

def get_latest_clean_run_id(cleaned_root):
    """获取 cleaned_jsonl 下最新的 run_id（按目录名排序）"""
    cleaned_dir = Path(cleaned_root)
    if not cleaned_dir.exists():
        return None
    run_dirs = [d for d in cleaned_dir.iterdir() if d.is_dir() and "_clean_" in d.name]
    if not run_dirs:
        return None
    run_dirs.sort(reverse=True)
    return run_dirs[0].name

def collect_kept_turns(cleaned_run_dir, logger):
    """
    从清洗结果目录中收集所有保留的 (id, turn)
    返回: kept = {dialog_id: set(turns)}
    """
    kept = defaultdict(set)
    if not cleaned_run_dir.exists():
        logger.error(f"清洗结果目录不存在: {cleaned_run_dir}")
        return kept
    for bucket_dir in cleaned_run_dir.iterdir():
        if not bucket_dir.is_dir():
            continue
        for jsonl_file in bucket_dir.glob("*.jsonl"):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        dialog_id = data.get('id')
                        turn = data.get('turn')
                        if dialog_id is not None and turn is not None:
                            kept[dialog_id].add(turn)
                    except json.JSONDecodeError:
                        logger.warning(f"{jsonl_file} 解析失败行: {line[:100]}")
    return kept

def apply_loss_to_original(original_dialogues, kept_turns, logger):
    """根据 kept_turns 修改原始对话中的 loss 字段"""
    total_assistant = 0
    total_true = 0
    for dialog_id, dialog in enumerate(original_dialogues):
        messages = dialog.get('messages', [])
        assistant_indices = []
        for idx, msg in enumerate(messages):
            if msg.get('role') == 'assistant':
                msg['loss'] = "False"
                assistant_indices.append(idx)
                total_assistant += 1
        for turn in kept_turns.get(dialog_id, set()):
            if turn < len(assistant_indices):
                msg_idx = assistant_indices[turn]
                messages[msg_idx]['loss'] = "True"
                total_true += 1
            else:
                logger.warning(f"对话 {dialog_id} 中 turn {turn} 超出范围 (共 {len(assistant_indices)} 个 assistant)")
    logger.info(f"统计: 总 assistant 消息数={total_assistant}, 保留(True)={total_true}, 丢弃(False)={total_assistant - total_true}")
    return original_dialogues

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, help="全局配置JSON字符串（优先级最高）")
    parser.add_argument("--source_run_id", type=str, help="清洗任务的 run_id（如 20250421_153022_clean_default）")
    parser.add_argument("--original_json", type=str, help="原始对话 JSON 路径")
    parser.add_argument("--cleaned_root", type=str, help="清洗结果根目录")
    parser.add_argument("--output_root", type=str, help="最终训练数据输出根目录")
    args = parser.parse_args()

    # ---------- 参数解析 ----------
    if args.config_json:
        config = json.loads(args.config_json)
        task_name = config['task_name']
        base_dir = Path(config['paths']['output']['base_dir'])
        task_dir = base_dir / task_name
        step_cfg = config.get('steps', {}).get('04_finalize', {})
        
        original_json = step_cfg.get('original_json') or (task_dir / "raw_dialogues.json")
        cleaned_root = step_cfg.get('cleaned_root') or (task_dir / "cleaned_jsonl")
        output_root = step_cfg.get('output_root') or (task_dir / "final_training_data")
        source_run_id = step_cfg.get('source_run_id')  # 可选，若不提供则自动取最新
        
        # 设置日志
        logger = setup_logger(task_dir, task_name)
        logger.info(f"任务名称: {task_name}")
        logger.info(f"任务目录: {task_dir}")
    else:
        # 独立命令行模式
        original_json = args.original_json
        cleaned_root = args.cleaned_root
        output_root = args.output_root
        source_run_id = args.source_run_id
        if not original_json or not cleaned_root or not output_root:
            print("错误：独立模式需要提供 --original_json, --cleaned_root, --output_root")
            sys.exit(1)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("Finalize")
        task_dir = Path(original_json).parent

    # 确定清洗结果目录
    cleaned_root_path = Path(cleaned_root)
    if source_run_id:
        run_id = source_run_id
        cleaned_dir = cleaned_root_path / run_id
        if not cleaned_dir.exists():
            logger.error(f"指定的清洗结果目录不存在: {cleaned_dir}")
            sys.exit(1)
    else:
        run_id = get_latest_clean_run_id(cleaned_root_path)
        if run_id is None:
            logger.error("未找到清洗结果目录，请先运行 03_clean_buckets_with_plots.py 或提供 --source_run_id")
            sys.exit(1)
        cleaned_dir = cleaned_root_path / run_id
        logger.info(f"自动选择最新的清洗结果: {run_id}")

    # 输出目录
    output_dir = Path(output_root) / f"{run_id}_final"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "cleaned_training_data.json"

    # 加载原始对话
    original_path = Path(original_json)
    if not original_path.exists():
        logger.error(f"原始对话文件不存在: {original_path}")
        sys.exit(1)
    logger.info(f"加载原始对话: {original_path}")
    with open(original_path, 'r', encoding='utf-8') as f:
        original_dialogues = json.load(f)
    logger.info(f"原始对话数量: {len(original_dialogues)}")

    # 收集保留的样本
    logger.info(f"扫描清洗结果: {cleaned_dir}")
    kept_turns = collect_kept_turns(cleaned_dir, logger)
    total_kept = sum(len(v) for v in kept_turns.values())
    logger.info(f"收集到 {len(kept_turns)} 个对话有保留轮次，总保留轮次数: {total_kept}")

    # 应用 loss 标记
    final_data = apply_loss_to_original(original_dialogues, kept_turns, logger)

    # 保存最终数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    logger.info(f"最终训练数据已保存: {output_file}")

    # 保存元数据
    metadata = {
        "run_id": f"{run_id}_final",
        "task": "finalize",
        "source_run_id": run_id,
        "source_cleaned_dir": str(cleaned_dir),
        "original_json": str(original_path),
        "output_file": str(output_file),
        "timestamp": datetime.now().isoformat(),
        "statistics": {
            "total_dialogues": len(original_dialogues),
            "total_assistant_messages": sum(1 for d in final_data for m in d['messages'] if m.get('role') == 'assistant'),
            "total_loss_true": sum(1 for d in final_data for m in d['messages'] if m.get('role') == 'assistant' and m.get('loss') == "True"),
        }
    }
    metadata_path = output_dir / "run_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"元数据已保存: {metadata_path}")

if __name__ == "__main__":
    main()