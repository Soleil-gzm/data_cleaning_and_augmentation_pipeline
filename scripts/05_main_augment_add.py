#!/usr/bin/env python3
"""
对话语义增强脚本（配置驱动版）
读取最终训练数据 JSON，对每个对话中的指定轮次进行多步叠加增强，
生成多个变体对话，输出：
  1. combined_augmented_xxx.json/jsonl : 原始+变体
  2. variants_only_xxx.json/jsonl     : 仅变体
支持两种配置方式：
  1) --config <yaml>           ：直接读取 pipeline YAML 配置，使用其中 steps.05_augment 段
  2) --config_json <json_str>   ：传入 JSON 字符串（兼容旧接口）
同时支持在 YAML 的 05_augment 段中通过 input_file 指定任意 JSON 输入路径，
并可用 --input_file 命令行参数覆盖配置中的输入文件。
支持通过配置中的 augment_weights 控制每种增强操作的相对概率。
支持 ASR 噪声增强（基于前置词和语义+拼音混合匹配）。

用 YAML 但临时换一个 JSON 文件:
python scripts/05_main_augment_add.py \
    --config configs/pipeline_config_v2.yaml \
    --input_file datas/my_custom_data.json
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

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def resolve_path(path_str, project_root, task_dir):
    """
    统一的路径解析工具：
      - 若 path_str 为 None 或空字符串，返回 None
      - 绝对路径直接返回
      - 包含 {task_dir} 占位符则替换为 task_dir
      - 其他相对路径视为相对于 project_root
    """
    if path_str is None:
        return None
    p = str(path_str).strip()
    if not p:
        return None
    if p.startswith("{task_dir}") or "{task_dir}" in p:
        p = p.format(task_dir=str(task_dir))
        return Path(p)
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return project_root / pp

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
        num_variants = max(1, min(3, len(enhanceable) // 2))

    aug_kwargs = config["augment_kwargs"]
    msg_prob = config.get("message_augment_prob", 1.0)   # 获取概率

    # 生成多个变体（外层循环）
    for var_id in range(num_variants):
        try:
            new_dialogue = deepcopy(original_dialogue)
            new_messages = new_dialogue["messages"]
            selected = enhanceable[:]  # 复制列表，所有可增强索引
            # 内层循环，负责遍历该对话副本中所有可增强的消息，并对每一条消息独立地进行增强（可能修改、可能跳过）。
            for idx in selected:
                # 按概率跳过增强
                if rng.random() > msg_prob:
                    logger.debug(f"  跳过消息 {idx}，不增强（概率未命中）")
                    continue

                original_text = new_messages[idx].get("content", "")
                if not original_text:
                    continue
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
    parser.add_argument("--config", type=str, help="Pipeline YAML 配置文件路径（读取其中 steps.05_augment 段）")
    parser.add_argument("--config_json", type=str, help="全局配置JSON字符串（兼容旧接口，优先级低于 --config）")
    parser.add_argument("--source_run_id", type=str, help="最终训练数据的 run_id")
    parser.add_argument("--input_file", type=str, help="输入 JSON 文件路径（可覆盖配置中的 input_file）")
    parser.add_argument("--output_dir", type=str, help="增强输出根目录（可覆盖配置中的 output_dir）")
    parser.add_argument("--num_variants", type=int, default=3)
    parser.add_argument("--min_turns", type=int, default=1)
    parser.add_argument("--max_turns", type=int, default=2)
    parser.add_argument("--target_roles", type=str, nargs='+', default=["user"])
    parser.add_argument("--only_loss_true", action="store_true")
    parser.add_argument("--adaptive_variants", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="default")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    # ---------- 配置模式 ----------
    # 优先级：--config (YAML) > --config_json (JSON 字符串) > 独立模式
    config = None
    step_cfg = {}

    if args.config:
        if not HAS_YAML:
            print("错误：需要 PyYAML 库来解析 YAML 配置，请先安装: pip install pyyaml")
            sys.exit(1)
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"错误：配置文件不存在: {config_path}")
            sys.exit(1)
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if config is None:
            print("错误：YAML 配置为空")
            sys.exit(1)
        step_cfg = config.get('steps', {}).get('05_augment', {}) or {}
        task_name = config.get('task_name', 'default_task')
        intermediate = config.get('paths', {}).get('intermediate', './intermediate')
        task_dir = project_root / intermediate / task_name

    elif args.config_json:
        config = json.loads(args.config_json)
        task_name = config['task_name']
        base_dir = Path(config['paths']['output']['base_dir'])
        task_dir = base_dir / task_name
        step_cfg = config.get('steps', {}).get('05_augment', {}) or {}

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
        augment_weights = None
        message_augment_prob = 1.0

        logger.info("=== 对话语义增强任务开始（独立模式）===")
        logger.info(f"输入文件: {input_file}")
        logger.info(f"输出目录: {output_dir}")

        enhance_config = {
            "num_variants_per_dialogue": num_variants,
            "min_enhance_turns": min_turns,
            "max_enhance_turns": max_turns,
            "target_roles": target_roles,
            "only_loss_true": only_loss_true,
            "adaptive_variants": adaptive_variants,
            "message_augment_prob": message_augment_prob,
            "augment_kwargs": {
                "num_variants": 1,
                "min_steps": 2,
                "max_steps": 3,
                "augment_weights": augment_weights
            }
        }

        _run_pipeline(input_file, output_dir, logger, enhance_config, args, tag, seed)
        return

    # ============ 配置模式（YAML 或 JSON 字符串）===========
    # ----- 读取基础参数 -----
    source_run_id = step_cfg.get('source_run_id') or args.source_run_id
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = step_cfg.get('tag', args.tag)

    # ----- 确定输出目录 -----
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
    else:
        output_dir = resolve_path(step_cfg.get('output_dir'), project_root, task_dir)
        if output_dir is None:
            output_dir = task_dir / "output_augmented_data" / f"{timestamp}_augment_{tag}"
        else:
            output_dir = output_dir / f"{timestamp}_augment_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    num_variants = step_cfg.get('num_variants', args.num_variants)
    min_turns = step_cfg.get('min_turns', args.min_turns)
    max_turns = step_cfg.get('max_turns', args.max_turns)
    target_roles = step_cfg.get('target_roles', args.target_roles)
    only_loss_true = step_cfg.get('only_loss_true', args.only_loss_true)
    adaptive_variants = step_cfg.get('adaptive_variants', args.adaptive_variants)
    seed = step_cfg.get('seed', args.seed)

    # ----- 读取增强操作权重 -----
    augment_weights = step_cfg.get('augment_weights', None)
    if augment_weights is not None:
        augment_weights = {k: float(v) for k, v in augment_weights.items()}
    else:
        augment_weights = {}

    # ----- 读取消息增强概率 -----
    message_augment_prob = step_cfg.get('message_augment_prob', 1.0)

    # ----- 确定输入文件（优先级：命令行 > 配置 input_file > source_run_id > 自动推断）-----
    input_file = None
    if args.input_file:
        input_file = resolve_path(args.input_file, project_root, task_dir)
    elif step_cfg.get('input_file'):
        input_file = resolve_path(step_cfg.get('input_file'), project_root, task_dir)
    elif source_run_id:
        input_file = task_dir / "final_training_data" / source_run_id / "training_data.json"
    else:
        final_root = task_dir / "final_training_data"
        latest_final_dir = get_latest_final_run_id(final_root)
        if latest_final_dir:
            input_file = final_root / latest_final_dir / "cleaned_training_data.json"

    if input_file is None or not Path(input_file).exists():
        print(f"错误：无法确定有效的输入 JSON 文件，请在 YAML 的 05_augment 段配置 input_file 或 source_run_id，"
              f"或使用命令行 --input_file 指定。当前解析到: {input_file}")
        sys.exit(1)

    # ----- 设置日志 -----
    logger = setup_logger(task_dir, f"{task_name}_augment")
    logger.info(f"任务名称: {task_name}")
    logger.info(f"项目根目录: {project_root}")
    logger.info(f"任务目录: {task_dir}")
    logger.info(f"输入文件: {input_file}")
    logger.info(f"增强输出目录: {output_dir}")
    logger.info(f"增强参数: num_variants={num_variants}, target_roles={target_roles}, "
                f"only_loss_true={only_loss_true}, seed={seed}")
    if augment_weights:
        logger.info(f"增强权重配置: {augment_weights}")
    else:
        logger.info("未提供增强权重，将使用均匀分布")

    # ----- 加载 ASR 增强器 -----
    asr_cache_cfg = step_cfg.get('asr_cache', {})
    if asr_cache_cfg:
        vectors_path = resolve_path(asr_cache_cfg.get('vectors_path'), project_root, task_dir)
        pinyin_path = resolve_path(asr_cache_cfg.get('pinyin_path'), project_root, task_dir)
        prev_map_path = resolve_path(asr_cache_cfg.get('prev_map_path'), project_root, task_dir)
        model_path = resolve_path(asr_cache_cfg.get('model_path'), project_root, task_dir)

        try:
            from common.asr_noise_augmenter import AsrNoiseAugmenter
            asr_augmenter = AsrNoiseAugmenter(
                vectors_path=str(vectors_path) if vectors_path else None,
                pinyin_path=str(pinyin_path) if pinyin_path else None,
                prev_map_path=str(prev_map_path) if prev_map_path and Path(prev_map_path).exists() else None,
                model_path=str(model_path) if model_path else None
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

    logger.debug(f"aug_utils.AUGMENT_FUNC_MAP 中的键: {list(aug_utils.AUGMENT_FUNC_MAP.keys())}")
    if 'asr_noise' not in aug_utils.AUGMENT_FUNC_MAP:
        logger.warning("警告: aug_utils.AUGMENT_FUNC_MAP 中未找到 'asr_noise'，ASR 增强将不可用！")

    enhance_config = {
        "num_variants_per_dialogue": num_variants,
        "min_enhance_turns": min_turns,
        "max_enhance_turns": max_turns,
        "target_roles": target_roles,
        "only_loss_true": only_loss_true,
        "adaptive_variants": adaptive_variants,
        "message_augment_prob": message_augment_prob,
        "augment_kwargs": {
            "num_variants": 1,
            "min_steps": 2,
            "max_steps": 3,
            "augment_weights": augment_weights
        }
    }

    _run_pipeline(input_file, output_dir, logger, enhance_config, args, tag, seed)


def _run_pipeline(input_file, output_dir, logger, enhance_config, args, tag, seed):
    """执行增强主流程：加载数据 → 逐对话增强 → 保存结果"""
    rng = random.Random(seed)

    logger.info("=== 对话语义增强任务开始 ===")
    
    aug_utils.reset_augment_perf_stats()

    # 加载原始数据
    logger.info("加载原始数据...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except Exception as e:
        logger.error(f"加载原始数据失败: {e}")
        sys.exit(1)
    logger.info(f"原始对话数量: {len(original_data)}")

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
        "augment_weights": enhance_config.get("augment_kwargs", {}).get("augment_weights"),
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
    
    aug_utils.print_augment_perf_stats()
    
    try:
        from common.asr_noise_augmenter import print_asr_global_stats
        print_asr_global_stats()
    except Exception:
        pass

if __name__ == "__main__":
    main()