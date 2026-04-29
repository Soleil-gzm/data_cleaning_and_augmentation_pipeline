#!/usr/bin/env python3
"""
应用清洗结果，生成最终训练 JSON
根据 cleaned_jsonl 中保留的 (id, turn) 对，将原始 JSON 中对应 assistant 的 loss 设为 True，其余设为 False。
支持自动获取最新清洗结果或手动指定 run_id。
输出目录：final_training_data/{source_run_id}_final/training_data.json
"""

import json
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ========== 配置 ==========
DEFAULT_ORIGINAL_JSON = "intermediate/raw_dialogues.json"
CLEANED_ROOT = "intermediate/output_cleaning/cleaned_jsonl"
OUTPUT_ROOT = "intermediate/output_cleaning/final_training_data"

def get_latest_clean_run_id():
    """获取 cleaned_jsonl 下最新的 run_id（按目录名排序）"""
    cleaned_dir = Path(CLEANED_ROOT)
    if not cleaned_dir.exists():
        return None
    # 只选取以 _clean_ 结尾的目录（即我们生成的 run_id）
    run_dirs = [d for d in cleaned_dir.iterdir() if d.is_dir() and "_clean_" in d.name]
    if not run_dirs:
        return None
    run_dirs.sort(reverse=True)  # 按名字降序，时间戳在开头，所以最新的在最前
    return run_dirs[0].name

def collect_kept_turns(cleaned_run_dir):
    """
    从清洗结果目录中收集所有保留的 (id, turn)
    cleaned_run_dir: Path 对象，例如 intermediate/output_cleaning/cleaned_jsonl/20250421_153022_clean_default/
    返回: kept = {dialog_id: set(turns)}
    """
    kept = defaultdict(set)
    # 遍历所有桶子目录
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
                        print(f"警告: {jsonl_file} 解析失败行: {line[:100]}")
    return kept

def apply_loss_to_original(original_dialogues, kept_turns):
    """根据 kept_turns 修改原始对话中的 loss 字段"""
    total_assistant = 0
    total_true = 0
    for dialog_id, dialog in enumerate(original_dialogues):
        messages = dialog.get('messages', [])
        assistant_indices = []
        # 先将所有 assistant 的 loss 设为 False
        for idx, msg in enumerate(messages):
            if msg.get('role') == 'assistant':
                msg['loss'] = "False"
                assistant_indices.append(idx)
                total_assistant += 1
        # 再将保留的 turn 设为 True
        for turn in kept_turns.get(dialog_id, set()):
            if turn < len(assistant_indices):
                msg_idx = assistant_indices[turn]
                messages[msg_idx]['loss'] = "True"
                total_true += 1
            else:
                print(f"警告: 对话 {dialog_id} 中 turn {turn} 超出范围 (共 {len(assistant_indices)} 个 assistant)")
    print(f"统计: 总 assistant 消息数 = {total_assistant}, 保留(True) = {total_true}, 丢弃(False) = {total_assistant - total_true}")
    return original_dialogues

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_run_id", type=str, default=None,
                        help="清洗任务的 run_id，例如 20250421_153022_clean_default")
    parser.add_argument("--original", type=str, default=DEFAULT_ORIGINAL_JSON,
                        help=f"原始对话 JSON 路径 (默认: {DEFAULT_ORIGINAL_JSON})")
    args = parser.parse_args()

    # 确定清洗结果目录
    if args.source_run_id:
        run_id = args.source_run_id
        cleaned_dir = Path(CLEANED_ROOT) / run_id
        if not cleaned_dir.exists():
            print(f"错误: 指定的清洗结果目录不存在: {cleaned_dir}")
            sys.exit(1)
    else:
        run_id = get_latest_clean_run_id()
        if run_id is None:
            print("错误: 未找到清洗结果目录，请先运行 03_clean_buckets_with_plots.py")
            sys.exit(1)
        cleaned_dir = Path(CLEANED_ROOT) / run_id
        print(f"自动选择最新的清洗结果: {run_id}")

    # 输出目录：final_training_data/{run_id}_final/
    output_dir = Path(OUTPUT_ROOT) / f"{run_id}_final"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "training_data.json"

    # 加载原始对话
    original_path = Path(args.original)
    if not original_path.exists():
        print(f"错误: 原始对话文件不存在: {original_path}")
        sys.exit(1)
    print(f"加载原始对话: {original_path}")
    with open(original_path, 'r', encoding='utf-8') as f:
        original_dialogues = json.load(f)
    print(f"原始对话数量: {len(original_dialogues)}")

    # 收集保留的样本
    print(f"扫描清洗结果: {cleaned_dir}")
    kept_turns = collect_kept_turns(cleaned_dir)
    print(f"收集到 {len(kept_turns)} 个对话有保留轮次，总保留轮次数: {sum(len(v) for v in kept_turns.values())}")

    # 应用 loss 标记
    final_data = apply_loss_to_original(original_dialogues, kept_turns)

    # 保存最终数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    print(f"最终训练数据已保存: {output_file}")

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
            "total_loss_true": sum(1 for d in final_data for m in d['messages'] if m.get('role') == 'assistant' and m.get('loss') == True),
        }
    }
    metadata_path = output_dir / "run_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"元数据已保存: {metadata_path}")

if __name__ == "__main__":
    main()


'''
# 自动选择最新清洗结果
python scripts/04_apply_cleaned_loss_direct.py

# 手动指定 run_id
python scripts/04_apply_cleaned_loss_direct.py --source_run_id 20260421_163719_clean_default
'''