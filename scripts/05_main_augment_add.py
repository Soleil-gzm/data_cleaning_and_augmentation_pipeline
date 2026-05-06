#!/usr/bin/env python3
"""
对话语义增强脚本（配置驱动版）
读取最终训练数据 JSON，对每个对话中的指定轮次进行多步叠加增强，
生成多个变体对话，输出新的 JSON/JSONL 文件（保留原始数据及 loss 标记）。
支持 --config_json 参数，统一日志，动态路径。
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

# ========== 日志配置 ==========
def setup_logger(task_dir, run_id):
    log_dir = task_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"augment_{run_id}.log"
    
    logger = logging.getLogger("Augment")
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

def get_latest_final_run_id(final_root):
    """获取 final_training_data 下最新的 *_final 目录名"""
    final_dir = Path(final_root)
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
        if only_loss_true and role == "assistant" and msg.get("loss") != "True":
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
            # 增强所有可增强的位置（可根据需求改为随机选择）
            selected = enhanceable[:]
            for idx in selected:
                original_text = new_messages[idx].get("content", "")
                if not original_text:
                    continue
                variants_list = aug_utils.augment_cell_multi(original_text, **aug_kwargs)
                if variants_list:
                    new_messages[idx]["content"] = variants_list[0]
            variants.append(new_dialogue)
        except Exception as e:
            logger.error(f"对话 {dialog_id} 生成变体 {var_id} 失败: {e}", exc_info=True)
            continue

    return variants

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, help="全局配置JSON字符串（优先级最高）")
    parser.add_argument("--source_run_id", type=str, help="最终训练数据的 run_id (例如 20250421_153022_clean_default_final)")
    parser.add_argument("--input_file", type=str, help="输入 JSON 文件路径（如果不使用 source_run_id）")
    parser.add_argument("--output_dir", type=str, help="增强输出根目录")
    parser.add_argument("--num_variants", type=int, default=3)
    parser.add_argument("--min_turns", type=int, default=1)
    parser.add_argument("--max_turns", type=int, default=2)
    parser.add_argument("--target_roles", type=str, nargs='+', default=["user"])
    parser.add_argument("--only_loss_true", action="store_true")
    parser.add_argument("--adaptive_variants", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="default", help="增强任务标签")
    args = parser.parse_args()

    # ---------- 参数解析 ----------
    if args.config_json:
        config = json.loads(args.config_json)
        task_name = config['task_name']
        base_dir = Path(config['paths']['output']['base_dir'])
        task_dir = base_dir / task_name
        step_cfg = config.get('steps', {}).get('05_augment', {})
        
        source_run_id = step_cfg.get('source_run_id') or args.source_run_id
        output_dir = step_cfg.get('output_dir') or Path(config['paths']['output']['final_output_dir']) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_augment_{step_cfg.get('tag', task_name)}"
        num_variants = step_cfg.get('num_variants', args.num_variants)
        min_turns = step_cfg.get('min_turns', args.min_turns)
        max_turns = step_cfg.get('max_turns', args.max_turns)
        target_roles = step_cfg.get('target_roles', args.target_roles)
        only_loss_true = step_cfg.get('only_loss_true', args.only_loss_true)
        adaptive_variants = step_cfg.get('adaptive_variants', args.adaptive_variants)
        seed = step_cfg.get('seed', args.seed)
        tag = step_cfg.get('tag', args.tag)
        
        # 确定输入文件
        if source_run_id:
            input_file = task_dir / "final_training_data" / source_run_id / "training_data.json"
        else:
            input_file = step_cfg.get('input_file') or (task_dir / "final_training_data" / get_latest_final_run_id(task_dir / "final_training_data") / "training_data.json")
        if not input_file or not Path(input_file).exists():
            logger.error(f"无法确定有效的输入文件，请提供 source_run_id 或 input_file")
            sys.exit(1)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        logger = setup_logger(task_dir, f"{task_name}_augment")
        logger.info(f"任务名称: {task_name}")
        logger.info(f"任务目录: {task_dir}")
    else:
        # 独立命令行模式
        input_file = args.input_file
        output_dir = args.output_dir
        source_run_id = args.source_run_id
        num_variants = args.num_variants
        min_turns = args.min_turns
        max_turns = args.max_turns
        target_roles = args.target_roles
        only_loss_true = args.only_loss_true
        adaptive_variants = args.adaptive_variants
        seed = args.seed
        tag = args.tag
        if not input_file and not source_run_id:
            print("错误：独立模式需要提供 --input_file 或 --source_run_id")
            sys.exit(1)
        if not output_dir:
            print("错误：独立模式需要提供 --output_dir")
            sys.exit(1)
        if source_run_id and not input_file:
            # 尝试根据 source_run_id 推断路径
            base = Path("intermediate/final_training_data")
            input_file = base / source_run_id / "training_data.json"
            if not input_file.exists():
                print(f"错误：根据 source_run_id 推断的文件不存在: {input_file}")
                sys.exit(1)
        input_file = Path(input_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # 独立模式简单日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("Augment")
        task_dir = input_file.parent.parent  # 推测

    rng = random.Random(seed)

    logger.info("=== 对话语义增强任务开始 ===")
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"增强参数: num_variants={num_variants}, min_turns={min_turns}, max_turns={max_turns}, target_roles={target_roles}, only_loss_true={only_loss_true}, adaptive_variants={adaptive_variants}, seed={seed}")

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
        "num_variants_per_dialogue": num_variants,
        "min_enhance_turns": min_turns,
        "max_enhance_turns": max_turns,
        "target_roles": target_roles,
        "only_loss_true": only_loss_true,
        "adaptive_variants": adaptive_variants,
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

    # # 分离原始数据和变体数据
    # original_count = len(original_data)
    # variants_only = all_dialogues[original_count:]   # 只取变体部分

    # # 1. 保存合并文件（原始 + 变体）
    # combined_json = output_dir / f"combined_augmented_{timestamp}.json"
    # combined_jsonl = output_dir / f"combined_augmented_{timestamp}.jsonl"

    # with open(combined_json, 'w', encoding='utf-8') as f:
    #     json.dump(all_dialogues, f, ensure_ascii=False, indent=2)
    # with open(combined_jsonl, 'w', encoding='utf-8') as f:
    #     for d in all_dialogues:
    #         f.write(json.dumps(d, ensure_ascii=False) + '\n')

    # # 2. 保存仅变体文件（只含增强生成的对话）
    # variants_json = output_dir / f"variants_only_{timestamp}.json"
    # variants_jsonl = output_dir / f"variants_only_{timestamp}.jsonl"

    # with open(variants_json, 'w', encoding='utf-8') as f:
    #     json.dump(variants_only, f, ensure_ascii=False, indent=2)
    # with open(variants_jsonl, 'w', encoding='utf-8') as f:
    #     for d in variants_only:
    #         f.write(json.dumps(d, ensure_ascii=False) + '\n')

    # logger.info(f"合并文件（原始+变体）已保存: {combined_json}, {combined_jsonl}")
    # logger.info(f"仅变体文件已保存: {variants_json}, {variants_jsonl}")

    # 保存元数据
    metadata = {
        "run_id": f"{timestamp}_augment_{tag}",
        "task": "augment",
        "source_file": str(input_file),
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
    print(f"\n增强完成！")
    print(f"  原始对话: {len(original_data)}")
    print(f"  生成变体: {total_variants}")
    print(f"  输出目录: {output_dir}")

if __name__ == "__main__":
    main()