#!/usr/bin/env python3
"""
对话语义增强脚本（配置驱动版）
读取最终训练数据 JSON，对每个对话中的指定轮次进行多步叠加增强，
生成多个变体对话，输出：
  1. combined_augmented_xxx.json/jsonl : 原始+变体
  2. variants_only_xxx.json/jsonl     : 仅变体
支持 --config_json 参数，统一日志，动态路径。
支持通过配置中的 augment_weights 控制每种增强操作的相对概率。
支持 ASR 噪声增强（基于前置词和语义+拼音混合匹配）。
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import augment_utils_add as aug_utils

# ========== 日志配置 ==========
def setup_logger(task_dir, run_id):
    log_dir = task_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"05_augment_{run_id}.log"
    
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
        if only_loss_true:
            loss_val = msg.get("loss")
            if isinstance(loss_val, str):
                loss_val = loss_val.lower() == "true"
            elif not isinstance(loss_val, bool):
                loss_val = False
            if not loss_val:
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

    logger.debug(f"对话 {dialog_id} 可增强位置索引: {enhanceable}")

    num_variants = config["num_variants_per_dialogue"]
    if config["adaptive_variants"]:
        num_variants = max(1, min(5, len(enhanceable) // 2))

    aug_kwargs = config["augment_kwargs"]

    for var_id in range(num_variants):
        try:
            new_dialogue = deepcopy(original_dialogue)
            new_messages = new_dialogue["messages"]
            selected = enhanceable[:]  # 全部选择（可改为随机选择部分）
            for idx in selected:
                original_text = new_messages[idx].get("content", "")
                if not original_text:
                    continue
                # 调用增强函数，传递 augment_weights
                variants_list = aug_utils.augment_cell_multi(original_text, **aug_kwargs)
                # 调试：记录返回的变体列表
                logger.debug(f"  原始文本: {original_text[:50]}...")
                logger.debug(f"  变体列表: {variants_list[:1] if variants_list else []}")
                if variants_list and variants_list[0] != original_text:
                    new_messages[idx]["content"] = variants_list[0]
                    logger.debug(f"  增强成功: [{original_text}] -> [{variants_list[0]}]")
                else:
                    logger.debug(f"  增强未产生变化: [{original_text}]")
            variants.append(new_dialogue)
        except Exception as e:
            logger.error(f"对话 {dialog_id} 生成变体 {var_id} 失败: {e}", exc_info=True)
            continue

    return variants

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, help="全局配置JSON字符串（优先级最高）")
    parser.add_argument("--source_run_id", type=str, help="最终训练数据的 run_id")
    parser.add_argument("--input_file", type=str, help="输入 JSON 文件路径")
    parser.add_argument("--output_dir", type=str, help="增强输出根目录")
    parser.add_argument("--num_variants", type=int, default=3)
    parser.add_argument("--min_turns", type=int, default=1)
    parser.add_argument("--max_turns", type=int, default=2)
    parser.add_argument("--target_roles", type=str, nargs='+', default=["user"])
    parser.add_argument("--only_loss_true", action="store_true")
    parser.add_argument("--adaptive_variants", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="default")
    args = parser.parse_args()

    # ---------- 配置模式 ----------
    if args.config_json:
        config = json.loads(args.config_json)
        task_name = config['task_name']
        base_dir = Path(config['paths']['output']['base_dir'])
        task_dir = base_dir / task_name
        step_cfg = config.get('steps', {}).get('05_augment', {})
        
        source_run_id = step_cfg.get('source_run_id') or args.source_run_id
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_output_dir = task_dir / "output_augmented_data" / f"{timestamp}_augment_{step_cfg.get('tag', task_name)}"
        output_dir_str = step_cfg.get('output_dir')
        if output_dir_str:
            output_dir = Path(output_dir_str)
            if not output_dir.is_absolute():
                output_dir = task_dir / output_dir
        else:
            output_dir = default_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        num_variants = step_cfg.get('num_variants', args.num_variants)
        min_turns = step_cfg.get('min_turns', args.min_turns)
        max_turns = step_cfg.get('max_turns', args.max_turns)
        target_roles = step_cfg.get('target_roles', args.target_roles)
        only_loss_true = step_cfg.get('only_loss_true', args.only_loss_true)
        adaptive_variants = step_cfg.get('adaptive_variants', args.adaptive_variants)
        seed = step_cfg.get('seed', args.seed)
        tag = step_cfg.get('tag', args.tag)
        
        # ----- 读取增强操作权重 -----
        augment_weights = step_cfg.get('augment_weights', None)
        if augment_weights is not None:
            augment_weights = {k: float(v) for k, v in augment_weights.items()}
        else:
            augment_weights = {}
        
        # ----- 确定输入文件 -----
        if source_run_id:
            input_file = task_dir / "final_training_data" / source_run_id / "training_data.json"
        else:
            input_file = step_cfg.get('input_file') or (task_dir / "final_training_data" / get_latest_final_run_id(task_dir / "final_training_data") / "cleaned_training_data.json")
        if not input_file or not Path(input_file).exists():
            print(f"错误：无法确定有效的输入文件，请提供 source_run_id 或 input_file")
            sys.exit(1)
        
        # ----- 设置日志 -----
        logger = setup_logger(task_dir, f"{task_name}_augment")
        logger.info(f"任务名称: {task_name}")
        logger.info(f"任务目录: {task_dir}")
        logger.info(f"增强输出目录: {output_dir}")
        if augment_weights:
            logger.info(f"增强权重配置: {augment_weights}")
        else:
            logger.info("未提供增强权重，将使用均匀分布")

        asr_cache_cfg = step_cfg.get('asr_cache', {})
        if asr_cache_cfg:
            # 项目根目录 = base_dir (intermediate) 的父目录
            project_root = base_dir.parent
            
            vectors_path = asr_cache_cfg.get('vectors_path')
            pinyin_path = asr_cache_cfg.get('pinyin_path')
            prev_map_path = asr_cache_cfg.get('prev_map_path')
            model_path = asr_cache_cfg.get('model_path')
            
            if vectors_path and not Path(vectors_path).is_absolute():
                vectors_path = project_root / vectors_path
            if pinyin_path and not Path(pinyin_path).is_absolute():
                pinyin_path = project_root / pinyin_path
            if prev_map_path and not Path(prev_map_path).is_absolute():
                prev_map_path = project_root / prev_map_path
            if model_path and not Path(model_path).is_absolute():
                model_path = project_root / model_path
                
            try:
                from common.asr_noise_augmenter import AsrNoiseAugmenter
                asr_augmenter = AsrNoiseAugmenter(
                    vectors_path=vectors_path,
                    pinyin_path=pinyin_path,
                    prev_map_path=prev_map_path if prev_map_path and Path(prev_map_path).exists() else None,
                    model_path=model_path
                )
                aug_utils.set_asr_augmenter(asr_augmenter)
                logger.info(f"ASR 增强器已加载，模型路径: {model_path}")
                logger.info(f"  异常词数量: {len(asr_augmenter.abnormal_words)}")
                logger.info(f"  前置词映射大小: {len(asr_augmenter.prev_to_abnormals)}")
                if len(asr_augmenter.prev_to_abnormals) > 0:
                    logger.debug(f"  前置词示例: {list(asr_augmenter.prev_to_abnormals.keys())[:10]}")
                else:
                    logger.warning("  前置词映射为空！请检查 prev_to_abnormals.pkl 文件是否正确生成。")
            except Exception as e:
                logger.warning(f"加载 ASR 增强器失败: {e}，将禁用 asr_noise 增强")
        else:
            logger.info("未配置 asr_cache，将禁用 asr_noise 增强")


        # ----- 调试：检查 AUGMENT_FUNC_MAP 是否包含 asr_noise -----
        logger.debug(f"aug_utils.AUGMENT_FUNC_MAP 中的键: {list(aug_utils.AUGMENT_FUNC_MAP.keys())}")
        if 'asr_noise' not in aug_utils.AUGMENT_FUNC_MAP:
            logger.warning("警告: aug_utils.AUGMENT_FUNC_MAP 中未找到 'asr_noise'，ASR 增强将不可用！")
    
    # ---------- 独立模式（不支持权重和 ASR 增强）----------
    else:
        if not args.input_file and not args.source_run_id:
            print("错误：独立模式需要提供 --input_file 或 --source_run_id")
            sys.exit(1)
        if not args.output_dir:
            print("错误：独立模式需要提供 --output_dir")
            sys.exit(1)
        if args.source_run_id and not args.input_file:
            base = Path("intermediate/final_training_data")
            input_file = base / args.source_run_id / "training_data.json"
            if not input_file.exists():
                print(f"错误：根据 source_run_id 推断的文件不存在: {input_file}")
                sys.exit(1)
        else:
            input_file = Path(args.input_file)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("Augment")
        task_dir = input_file.parent.parent
        num_variants = args.num_variants
        min_turns = args.min_turns
        max_turns = args.max_turns
        target_roles = args.target_roles
        only_loss_true = args.only_loss_true
        adaptive_variants = args.adaptive_variants
        seed = args.seed
        tag = args.tag
        augment_weights = None   # 独立模式不支持权重
        # 独立模式下不加载 ASR 增强器，因为需要配置文件提供参数

    rng = random.Random(seed)

    logger.info("=== 对话语义增强任务开始 ===")
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"增强参数: num_variants={num_variants}, min_turns={min_turns}, max_turns={max_turns}, target_roles={target_roles}, only_loss_true={only_loss_true}, adaptive_variants={adaptive_variants}, seed={seed}")
    if augment_weights:
        logger.info(f"使用自定义增强权重: {augment_weights}")

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
    enhance_config = {
        "num_variants_per_dialogue": num_variants,
        "min_enhance_turns": min_turns,
        "max_enhance_turns": max_turns,
        "target_roles": target_roles,
        "only_loss_true": only_loss_true,
        "adaptive_variants": adaptive_variants,
        "augment_kwargs": {
            "num_variants": 1,
            "min_steps": 2,
            "max_steps": 3,
            "augment_weights": augment_weights   # 传递权重配置（独立模式下为 None）
        }
    }

    # 分别收集原始和变体
    all_original = []
    all_variants = []
    total_variants = 0
    failed_dialogues = []

    for idx, dialogue in enumerate(original_data):
        all_original.append(dialogue)
        try:
            variants = enhance_dialogue(dialogue, enhance_config, rng, logger, idx)
            all_variants.extend(variants)
            total_variants += len(variants)
        except Exception as e:
            logger.error(f"对话 {idx} 增强过程出现未捕获异常: {e}", exc_info=True)
            failed_dialogues.append(idx)
            continue

        if (idx + 1) % 100 == 0:
            logger.debug(f"已处理 {idx+1}/{len(original_data)} 个对话，生成 {total_variants} 个变体")

    # 合并所有对话
    all_dialogues = all_original + all_variants
    logger.info(f"增强完成: 原始 {len(original_data)}，变体 {total_variants}，总计 {len(all_dialogues)}")
    if failed_dialogues:
        logger.warning(f"有 {len(failed_dialogues)} 个对话增强失败: {failed_dialogues[:10]}{'...' if len(failed_dialogues)>10 else ''}")

    # 保存文件
    save_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    combined_json = output_dir / f"combined_augmented_{save_timestamp}.json"
    combined_jsonl = output_dir / f"combined_augmented_{save_timestamp}.jsonl"

    with open(combined_json, 'w', encoding='utf-8') as f:
        json.dump(all_dialogues, f, ensure_ascii=False, indent=2)

    with open(combined_jsonl, 'w', encoding='utf-8') as f:
        for d in all_dialogues:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    variants_json = output_dir / f"variants_only_{save_timestamp}.json"
    variants_jsonl = output_dir / f"variants_only_{save_timestamp}.jsonl"

    with open(variants_json, 'w', encoding='utf-8') as f:
        json.dump(all_variants, f, ensure_ascii=False, indent=2)
    with open(variants_jsonl, 'w', encoding='utf-8') as f:
        for d in all_variants:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    logger.info(f"合并文件（原始+变体）已保存: {combined_json}, {combined_jsonl}")
    logger.info(f"仅变体文件已保存: {variants_json}, {variants_jsonl}")

    # 元数据
    metadata = {
        "run_id": f"{save_timestamp}_augment_{tag}",
        "task": "augment",
        "source_file": str(input_file),
        "command_line": " ".join(sys.argv),
        "config": enhance_config,
        "augment_weights": augment_weights,
        "statistics": {
            "original_dialogues": len(original_data),
            "generated_variants": total_variants,
            "total_dialogues": len(all_dialogues),
            "failed_dialogues": failed_dialogues
        },
        "output_files": {
            "combined": [str(combined_json), str(combined_jsonl)],
            "variants_only": [str(variants_json), str(variants_jsonl)]
        }
    }
    metadata_path = output_dir / "run_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"增强任务完成，结果保存在: {output_dir}")
    print(f"\n增强完成！")
    print(f"  原始对话: {len(original_data)}")
    print(f"  生成变体: {total_variants}")
    print(f"  输出目录: {output_dir}")

if __name__ == "__main__":
    main()