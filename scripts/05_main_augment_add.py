#!/usr/bin/env python3
"""
对话语义增强脚本（基于清洗后的 JSON）
读取最终训练数据 JSON，对每个对话中的指定轮次进行多步叠加增强，
生成多个变体对话，输出新的 JSON/JSONL 文件（保留原始数据及 loss 标记）。

日志机制：
- 控制台只输出 INFO 及以上级别（任务开始、完成、统计汇总）
- 详细 DEBUG 信息（每100个对话进度、保存文件路径等）写入日志文件
- 增强过程中的异常会记录到日志，但不会中断任务
"""

import json
import argparse
import random
import sys
import os
import logging
from copy import deepcopy
from pathlib import Path
from datetime import datetime

# 导入增强工具包（请确保路径正确）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import augment_utils_add as aug_utils

# ========== 配置 ==========
DEFAULT_INPUT_ROOT = "intermediate/output_cleaning/final_training_data"
OUTPUT_ROOT = "output_augmented_data"
LOG_ROOT = "intermediate/logs_augmentation"          # 日志文件统一存放目录

def setup_logger(log_dir, run_id):
    """配置日志：文件记录 DEBUG 及以上，控制台只记录 INFO 及以上"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"augment_{run_id}.log"

    logger = logging.getLogger("DialogueAugment")
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    # 文件处理器：记录所有级别
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)

    # 控制台处理器：只记录 INFO 及以上
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def get_latest_final_run_id():
    """获取 final_training_data 下最新的 *_final 目录名"""
    final_dir = Path(DEFAULT_INPUT_ROOT)
    if not final_dir.exists():
        return None
    dirs = [d for d in final_dir.iterdir() if d.is_dir() and d.name.endswith("_final")]
    if not dirs:
        return None
    dirs.sort(reverse=True)
    return dirs[0].name

def get_enhanceable_indices(messages, target_roles, only_loss_true):
    """返回可以增强的消息索引列表"""
    indices = []
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role not in target_roles:
            continue
        content = msg.get("content", "")
        if not content.strip():
            continue
        if only_loss_true and role == "user" and msg.get("loss") != True:
            continue
        indices.append(idx)
    return indices

def enhance_dialogue(original_dialogue, config, rng, logger, dialog_id):
    """生成变体对话列表，异常捕获并记录日志"""
    variants = []
    messages = original_dialogue.get("messages", [])
    if not messages:
        logger.debug(f"对话 {dialog_id} 无 messages，跳过")
        return variants

    try:
        enhanceable = get_enhanceable_indices(messages, config["target_roles"], config["only_loss_true"])
    except Exception as e:
        logger.error(f"对话 {dialog_id} 获取可增强索引失败: {e}")
        return variants

    if not enhanceable:
        logger.debug(f"对话 {dialog_id} 无可增强轮次")
        return variants

    num_variants = config["num_variants_per_dialogue"]
    if config["adaptive_variants"]:
        num_variants = max(1, min(5, len(enhanceable) // 2))

    min_turns = config["min_enhance_turns"]
    max_turns = config["max_enhance_turns"]
    aug_kwargs = config["augment_kwargs"]

    for var_id in range(num_variants):
        try:
            new_dialogue = deepcopy(original_dialogue)
            new_messages = new_dialogue["messages"]

            # k = rng.randint(min_turns, max_turns)
            # if k > len(enhanceable):
            #     k = len(enhanceable)
            # selected = rng.sample(enhanceable, k)

            # 增强所有可增强的位置
            selected = enhanceable[:]   # 全部选择

            for idx in selected:
                original_text = new_messages[idx].get("content", "")
                if not original_text:
                    continue
                variants_list = aug_utils.augment_cell_multi(original_text, **aug_kwargs)
                if variants_list:
                    new_messages[idx]["content"] = variants_list[0]
            # 添加元数据
            # new_dialogue["_augmented_from"] = original_dialogue.get("id", None)
            # new_dialogue["_variant_id"] = var_id
            variants.append(new_dialogue)
        except Exception as e:
            logger.error(f"对话 {dialog_id} 生成变体 {var_id} 失败: {e}", exc_info=True)
            continue

    return variants

def main():
    parser = argparse.ArgumentParser()
    ''' 语义增强参数运行: python scripts/05_main_augment_add.py --tag <augment_tag> [options]

    # 只增强 loss=True 的 assistant，生成 5 个变体
        python scripts/06_augment_dialogues.py --tag lossOnly --only_loss_true --num_variants 5

        # 自适应变体数量，增强所有角色
        python scripts/06_augment_dialogues.py --tag adaptive_all --adaptive_variants 
    '''
    parser.add_argument("--source_run_id", type=str, default=None,
                        help="最终训练数据的 run_id (例如 20250421_153022_clean_default_final)")
    parser.add_argument("--tag", type=str, default="default", help="增强任务标签")
    parser.add_argument("--num_variants", type=int, default=3, help="每个原始对话生成的变体数量")
    parser.add_argument("--min_turns", type=int, default=1, help="每个变体中最少增强轮次数")
    parser.add_argument("--max_turns", type=int, default=2, help="每个变体中最少增强轮次数")
    parser.add_argument("--target_roles", type=str, nargs='+', default=["user"],
                        help="要增强的角色，可选 user/assistant")
    parser.add_argument("--only_loss_true", action="store_true",
                        help="是否只增强 loss=True 的 user 消息")
    parser.add_argument("--adaptive_variants", action="store_true",
                        help="根据可增强轮次数自动调整变体数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 定位输入文件
    if args.source_run_id:
        input_dir = Path(DEFAULT_INPUT_ROOT) / args.source_run_id
        if not input_dir.exists():
            print(f"错误: 指定的最终数据目录不存在: {input_dir}")
            sys.exit(1)
        input_file = input_dir / "training_data.json"
    else:
        run_id = get_latest_final_run_id()
        if run_id is None:
            print("错误: 未找到最终训练数据，请先运行 04_apply_cleaned_loss_direct.py")
            sys.exit(1)
        input_file = Path(DEFAULT_INPUT_ROOT) / run_id / "training_data.json"
        print(f"自动选择最新数据: {run_id}")

    if not input_file.exists():
        print(f"错误: 输入文件不存在: {input_file}")
        sys.exit(1)

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_augment_{args.tag}"
    output_dir = Path(OUTPUT_ROOT) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # 配置日志（日志文件统一放在 LOG_ROOT 下）
    logger = setup_logger(LOG_ROOT, run_id)
    logger.info("=== 对话语义增强任务开始 ===")
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"增强参数: {vars(args)}")

    # 加载原始数据
    logger.info("加载原始数据...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except Exception as e:
        logger.error(f"加载原始数据失败: {e}")
        sys.exit(1)
    logger.info(f"原始对话数量: {len(original_data)}")

    # 增强配置
    config = {
        "num_variants_per_dialogue": args.num_variants,
        "min_enhance_turns": args.min_turns,
        "max_enhance_turns": args.max_turns,
        "target_roles": args.target_roles,
        "only_loss_true": args.only_loss_true,
        "adaptive_variants": args.adaptive_variants,
        "augment_kwargs": {
            "num_variants": 1,
            "min_steps": 2,
            "max_steps": 3
        }
    }

    all_dialogues = []
    total_variants = 0
    failed_dialogues = []

    for idx, dialogue in enumerate(original_data):
        all_dialogues.append(dialogue)  # 保留原始
        try:
            variants = enhance_dialogue(dialogue, config, rng, logger, idx)
            all_dialogues.extend(variants)
            total_variants += len(variants)
        except Exception as e:
            logger.error(f"对话 {idx} 增强过程出现未捕获异常: {e}", exc_info=True)
            failed_dialogues.append(idx)
            continue

        if (idx + 1) % 100 == 0:
            logger.debug(f"已处理 {idx+1}/{len(original_data)} 个对话，生成 {total_variants} 个变体")

    logger.info(f"增强完成: 原始 {len(original_data)}，变体 {total_variants}，总计 {len(all_dialogues)}")
    if failed_dialogues:
        logger.warning(f"有 {len(failed_dialogues)} 个对话增强失败: {failed_dialogues[:10]}{'...' if len(failed_dialogues)>10 else ''}")

    # 保存 JSON
    output_json = output_dir / f"augmented_data_{timestamp}.json"
    logger.debug(f"保存 JSON 文件: {output_json}")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_dialogues, f, ensure_ascii=False, indent=2)

    # 保存 JSONL
    output_jsonl = output_dir / f"augmented_data_{timestamp}.jsonl"
    logger.debug(f"保存 JSONL 文件: {output_jsonl}")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for d in all_dialogues:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    # 保存元数据
    metadata = {
        "run_id": run_id,
        "task": "augment",
        "source_run_id": input_file.parent.name,
        "source_path": str(input_file),
        "command_line": " ".join(sys.argv),
        "config": config,
        "statistics": {
            "original_dialogues": len(original_data),
            "generated_variants": total_variants,
            "total_dialogues": len(all_dialogues),
            "failed_dialogues": failed_dialogues
        },
        "output_files": [str(output_json), str(output_jsonl)]
    }
    metadata_path = output_dir / "run_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"增强任务完成，结果保存在: {output_dir}")
    # 控制台输出简洁信息
    print(f"\n增强完成！")
    print(f"  原始对话: {len(original_data)}")
    print(f"  生成变体: {total_variants}")
    print(f"  输出目录: {output_dir}")

if __name__ == "__main__":
    main()