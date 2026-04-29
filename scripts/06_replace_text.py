#!/usr/bin/env python3
"""
文本替换脚本（含 loss 字段类型转换）
将生成的增强数据中的特定词语进行替换：
    1. "洋钱罐平台" -> "华夏银行"
    2. "洋钱罐" -> "华夏"
    3. "平台" -> "银行"
并将 loss 字段从布尔值转换为字符串 "True"/"False"。
支持自动获取最新 JSON 文件或手动指定输入文件。
输出文件保存在同目录下，文件名添加 _replaced 后缀。
"""

import json
import os
import sys
import argparse
from pathlib import Path

# ========== 配置 ==========
DEFAULT_OUTPUT_SUFFIX = "_replaced"
OUTPUT_BASE = "output_augmented_data"

def find_latest_augmented_json():
    """在 OUTPUT_BASE 目录下查找最新的 augmented_data_*.json 文件"""
    base_dir = Path(OUTPUT_BASE)
    if not base_dir.exists():
        return None
    
    candidates = []
    for subdir in base_dir.iterdir():
        if not subdir.is_dir():
            continue
        for json_file in subdir.glob("augmented_data_*.json"):
            candidates.append(json_file)
    
    if not candidates:
        return None
    
    # 按文件修改时间排序，取最新的
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest

def apply_replacements(text):
    """对文本按顺序执行替换"""
    if not isinstance(text, str):
        return text
    # 顺序很重要：先替换长串，再替换短串，最后替换单个词
    text = text.replace("洋钱罐平台", "华夏银行")
    text = text.replace("洋钱罐", "华夏")
    text = text.replace("平台", "银行")
    return text

def convert_loss_to_string(messages):
    """
    将 messages 中每个消息的 loss 字段从布尔值转换为字符串 "True"/"False"
    如果 loss 不存在则跳过，如果已经是字符串则不做修改
    """
    for msg in messages:
        if "loss" in msg:
            loss_val = msg["loss"]
            if isinstance(loss_val, bool):
                msg["loss"] = "True" if loss_val else "False"
    return messages

def process_messages(messages, stats):
    """处理单个对话中的 messages 列表（先替换文本，再转换 loss 类型）"""
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if content and isinstance(content, str):
            original = content
            new_content = apply_replacements(original)
            if new_content != original:
                stats["total_replacements"] += 1
                if "洋钱罐平台" in original:
                    stats["replace_platform_yqg"] += original.count("洋钱罐平台")
                if "洋钱罐" in original and "洋钱罐平台" not in original:
                    stats["replace_yqg"] += original.count("洋钱罐")
                if "平台" in original:
                    stats["replace_platform"] += original.count("平台")
                msg["content"] = new_content
    return messages

def main():
    parser = argparse.ArgumentParser(description="替换增强数据中的特定词语并转换 loss 类型")
    parser.add_argument("--input", type=str, default=None,
                        help="指定输入的 JSON 文件路径（若未提供则自动查找最新文件）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径（默认在输入文件同目录下添加 _replaced 后缀）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅统计替换次数，不实际写入文件")
    args = parser.parse_args()

    # 确定输入文件
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"错误: 指定的输入文件不存在: {input_path}")
            sys.exit(1)
    else:
        input_path = find_latest_augmented_json()
        if input_path is None:
            print("错误: 未找到任何 augmented_data_*.json 文件，请先运行增强脚本。")
            sys.exit(1)
        print(f"自动选择最新增强文件: {input_path}")

    # 加载数据
    print(f"加载文件: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"共加载 {len(data)} 条对话")

    # 统计信息
    stats = {
        "total_replacements": 0,
        "replace_platform_yqg": 0,
        "replace_yqg": 0,
        "replace_platform": 0
    }

    # 处理每条对话
    for idx, dialogue in enumerate(data):
        messages = dialogue.get("messages", [])
        if messages:
            # 第一步：文本替换
            process_messages(messages, stats)
            # 第二步：转换 loss 类型（布尔 -> 字符串）
            convert_loss_to_string(messages)
        if (idx + 1) % 1000 == 0:
            print(f"已处理 {idx+1} 条对话...")

    # 输出统计
    print("\n替换统计:")
    print(f"  '洋钱罐平台' -> '华夏银行' 次数: {stats['replace_platform_yqg']}")
    print(f"  '洋钱罐' -> '华夏' 次数: {stats['replace_yqg']}")
    print(f"  '平台' -> '银行' 次数: {stats['replace_platform']}")
    print(f"  总替换操作次数: {stats['total_replacements']}")

    # 可选：统计 loss 转换情况
    loss_true_count = 0
    loss_false_count = 0
    for dialogue in data:
        for msg in dialogue.get("messages", []):
            if "loss" in msg and msg["loss"] == "True":
                loss_true_count += 1
            elif "loss" in msg and msg["loss"] == "False":
                loss_false_count += 1
    print(f"\nLoss 字段转换: True={loss_true_count}, False={loss_false_count}")

    if args.dry_run:
        print("\n[干运行] 未写入文件。")
        return

    # 确定输出文件路径
    if args.output:
        output_path = Path(args.output)
    else:
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}{DEFAULT_OUTPUT_SUFFIX}.json"
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n替换后的文件已保存至: {output_path}")

if __name__ == "__main__":
    main()