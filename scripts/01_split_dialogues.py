#!/usr/bin/env python3
"""
多轮对话拆分脚本（配置驱动版）
将原始 JSON 文件中的每个对话按轮次拆分成样本，保存为 JSONL 文件，并统计轮次分布。
支持流式读取、分批输出、断点续传。

参数优先级：
  1. --config_json（最高，用于流水线集成）
  2. 独立命令行参数（便于单独调试）
  3. 默认硬编码值（向后兼容，会打印警告）
"""

import json
import os
import sys
import argparse
import logging
from collections import defaultdict
from pathlib import Path
import ijson
from tqdm import tqdm

# ========== 辅助函数 ==========
def setup_logger(log_dir, task_name):
    """配置日志：文件记录 DEBUG，控制台 INFO"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"01_split_dialogues_{task_name}.log"
    
    logger = logging.getLogger("SplitDialogues")
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

def get_last_processed_index(progress_file):
    """读取上次处理到的对话索引"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return -1
    return -1

def update_progress(progress_file, index):
    """更新进度文件"""
    with open(progress_file, 'w') as f:
        f.write(str(index))

def get_output_filename(batch_start, batch_end):
    """根据对话索引范围生成输出文件名"""
    return f"sample_{batch_start:08d}_{batch_end:08d}.jsonl"

def process_dialog(dialog_id, messages, turn_counter):
    """
    处理单个对话，拆分成多个样本（每个样本对应一轮 assistant 回复）
    返回样本列表
    """
    samples = []
    history_pairs = []   # 保存之前所有轮的 (user_raw, assistant_raw)
    pending_user = None
    # 从索引1开始，跳过 system
    for i, msg in enumerate(messages):
        role = msg.get('role')
        content = msg.get('content', '')
        if role == 'user':
            pending_user = msg
        elif role == 'assistant' and pending_user is not None:
            turn = len(samples)
            user_raw = pending_user.get('content', '')
            assistant_raw = msg.get('content', '')
            
            # 构建历史文本（原始内容，不加前缀）
            history_text = ""
            for hist_user_raw, hist_assistant_raw in history_pairs:
                history_text += f"{hist_user_raw}\n{hist_assistant_raw}\n"
            
            current_input = user_raw
            if history_text:
                full_input = f"Q：{history_text}{current_input}"
            else:
                full_input = f"Q：{current_input}" if current_input else "Q："
            
            target_output = f"A：{assistant_raw}" if assistant_raw else "A："
            full_text = history_text + f"{user_raw}\n{assistant_raw}"
            
            sample = {
                "id": dialog_id,
                "turn": turn,
                "user_input": full_input,
                "target_output": target_output,
                "loss": msg.get('loss', False),
                "text": full_text
            }
            samples.append(sample)
            turn_counter[turn] += 1
            history_pairs.append((user_raw, assistant_raw))
            pending_user = None
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, help="全局配置JSON字符串（优先级最高）")
    parser.add_argument("--input_json", type=str, help="原始 JSON 文件路径")
    parser.add_argument("--output_dir", type=str, help="样本输出目录")
    parser.add_argument("--stats_dir", type=str, help="统计信息输出目录")
    parser.add_argument("--batch_size", type=int, default=120000, help="每个 JSONL 文件包含的对话数")
    parser.add_argument("--progress_file", type=str, help="进度文件路径（可选，默认在 stats_dir 下）")
    args = parser.parse_args()

    # ---------- 参数解析 ----------
    if args.config_json:
        config = json.loads(args.config_json)
        task_name = config['task_name']
        base_dir = Path(config['paths']['output']['base_dir'])
        task_dir = base_dir / task_name
        
        # 从配置中读取步骤01的参数
        step_cfg = config.get('steps', {}).get('01_split', {})
        input_json = step_cfg.get('input_json') or (task_dir / "raw_dialogues.json")
        output_dir = step_cfg.get('output_dir') or (task_dir / "samples")
        stats_dir = step_cfg.get('stats_dir') or (task_dir / "stats")
        batch_size = step_cfg.get('batch_size', args.batch_size)
        progress_file = step_cfg.get('progress_file') or (stats_dir / "progress.txt")
        
        # 设置日志
        log_dir = config.get('logging', {}).get('log_dir') or (task_dir / "logs")
        logger = setup_logger(log_dir, task_name)
        logger.info(f"任务名称: {task_name}")
        logger.info(f"任务目录: {task_dir}")
    else:
        # 独立命令行模式
        input_json = args.input_json
        output_dir = args.output_dir
        stats_dir = args.stats_dir
        batch_size = args.batch_size
        progress_file = args.progress_file
        if not input_json or not output_dir or not stats_dir:
            print("警告: 未提供 --config_json 且缺少必需的独立参数，将使用硬编码默认值")
            input_json = input_json or "intermediate/raw_dialogues.json"
            output_dir = output_dir or "intermediate/samples"
            stats_dir = stats_dir or "intermediate/stats"
            batch_size = batch_size
        # 独立模式简单日志（只输出到控制台）
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("SplitDialogues")
        task_name = Path(input_json).stem  # 临时任务名
        if not progress_file:
            progress_file = Path(stats_dir) / "progress.txt"

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(stats_dir).mkdir(parents=True, exist_ok=True)
    progress_file = Path(progress_file)

    # 读取进度
    last_idx = get_last_processed_index(progress_file)
    start_idx = last_idx + 1

    # 检查输入文件是否存在
    if not os.path.exists(input_json):
        logger.error(f"输入文件不存在: {input_json}")
        sys.exit(1)

    # 获取总对话数（快速扫描）
    logger.info("正在统计总对话数...")
    with open(input_json, 'rb') as f:
        total_dialogues = sum(1 for _ in ijson.items(f, 'item'))
    logger.info(f"总对话数: {total_dialogues}")

    # 如果进度已超出，重置
    if start_idx >= total_dialogues:
        logger.warning(f"上次处理索引 {last_idx} 已达或超过总对话数 {total_dialogues}，重置进度文件，从头开始。")
        if progress_file.exists():
            progress_file.unlink()
        start_idx = 0
        last_idx = -1

    # 统计计数器
    turn_counter = defaultdict(int)

    # 批次控制
    batch_start = start_idx
    batch_end = start_idx - 1
    current_file = None
    current_file_path = None
    current_file_count = 0

    # 流式处理
    with open(input_json, 'rb') as f:
        items = ijson.items(f, 'item')
        pbar = tqdm(desc="Processing dialogues", unit="dialogue")
        processed = 0
        current_idx = 0
        for dialog in items:
            if current_idx < start_idx:
                current_idx += 1
                continue

            dialog_id = current_idx
            messages = dialog.get('messages', [])
            if not messages:
                current_idx += 1
                processed += 1
                pbar.update(1)
                continue

            samples = process_dialog(dialog_id, messages, turn_counter)

            # 写入当前批次文件
            if current_file is None or current_file_count >= batch_size:
                if current_file:
                    current_file.close()
                batch_start = current_idx
                batch_end = batch_start + batch_size - 1
                output_filename = get_output_filename(batch_start, batch_end)
                current_file_path = Path(output_dir) / output_filename
                current_file = open(current_file_path, 'a', encoding='utf-8')
                current_file_count = 0
                logger.info(f"创建新批次文件: {output_filename}")

            for sample in samples:
                current_file.write(json.dumps(sample, ensure_ascii=False) + '\n')
                current_file_count += 1

            current_idx += 1
            processed += 1
            if processed % 1000 == 0:
                update_progress(progress_file, current_idx - 1)
            pbar.update(1)

        if current_file:
            current_file.close()
        pbar.close()

    # 最终更新进度
    update_progress(progress_file, current_idx - 1)

    # 保存统计结果
    stats = {
        "total_samples": sum(turn_counter.values()),
        "turn_distribution": dict(turn_counter)
    }
    stats_path = Path(stats_dir) / "turn_distribution.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"统计结果已保存到 {stats_path}")

    # 打印摘要
    logger.info("\n轮次分布摘要：")
    for turn, cnt in sorted(turn_counter.items()):
        logger.info(f"  第 {turn} 轮: {cnt} 条样本")
    logger.info(f"总样本数: {stats['total_samples']}")

if __name__ == "__main__":
    main()