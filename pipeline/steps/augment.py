"""
05_augment：语义增强（使用新增强器架构）
完全替代 common/，所有增强逻辑由 pipeline.augmenters 提供
"""
import json
import random
import sys
import os
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from ..core.step import PipelineStep
from ..augmenters import CompositeAugmenter, AugmenterRegistry

# 导入以确保所有增强器已注册
from ..augmenters import *  # noqa


class AugmentStep(PipelineStep):
    def run(self) -> bool:
        cfg = self.context.get_step_config("05_augment")

        # 确定输入文件
        source_run_id = cfg.get("source_run_id")
        input_file = cfg.get("input_file")

        if input_file:
            input_path = self.context.resolve_path(input_file)
        elif source_run_id:
            final_root = self.context.task_dir / "final_training_data"
            input_path = final_root / source_run_id / "cleaned_training_data.json"
        else:
            final_root = self.context.task_dir / "final_training_data"
            latest_final_dir = self._get_latest_final_dir(final_root)
            if latest_final_dir:
                input_path = final_root / latest_final_dir / "cleaned_training_data.json"
            else:
                self.logger.error("无法确定输入文件，请配置 source_run_id 或 input_file")
                return False

        if not input_path.exists():
            self.logger.error(f"输入文件不存在: {input_path}")
            return False

        # 生成 run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = cfg.get("tag", "augment")
        run_id = f"{timestamp}_augment_{tag}"

        # 输出目录
        output_base = cfg.get("output_dir") or (self.context.task_dir / "output_augmented_data")
        output_dir = Path(output_base) / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # 并行配置
        global_workers = self.context.config.get("executor", {}).get("max_workers", 1)
        max_workers = cfg.get("max_workers", global_workers)

        # 参数
        num_variants = cfg.get("num_variants", 3)
        target_roles = cfg.get("target_roles", ["user"])
        only_loss_true = cfg.get("only_loss_true", True)
        adaptive_variants = cfg.get("adaptive_variants", False)
        seed = cfg.get("seed", 42)
        message_augment_prob = cfg.get("message_augment_prob", 1.0)

        # 构建增强器配置
        augmenters_cfg = cfg.get("augmenters", {})

        # 兼容旧格式：如果使用 augment_weights，转换为新格式
        if not augmenters_cfg:
            augment_weights = cfg.get("augment_weights", {})
            for name, weight in augment_weights.items():
                if weight > 0:
                    augmenters_cfg[name] = {"enabled": True, "weight": weight}

        composite_config = {"augmenters": augmenters_cfg}

        self.logger.info(f"增强 run_id: {run_id}")
        self.logger.info(f"输入: {input_path}")
        self.logger.info(f"输出: {output_dir}")
        self.logger.info(f"启用增强器: {list(augmenters_cfg.keys())}")
        if max_workers > 1:
            self.logger.info(f"并行模式，进程数: {max_workers}")

        # 加载数据
        with open(input_path, "r", encoding="utf-8") as f:
            original_data = json.load(f)
        self.logger.info(f"原始对话数: {len(original_data)}")

        enhance_config = {
            "num_variants_per_dialogue": num_variants,
            "target_roles": target_roles,
            "only_loss_true": only_loss_true,
            "adaptive_variants": adaptive_variants,
            "message_augment_prob": message_augment_prob,
            "composite_config": composite_config,
            "seed": seed
        }

        # 执行增强
        if max_workers <= 1:
            all_original, all_variants, total_variants, failed = self._run_serial(original_data, enhance_config)
        else:
            all_original, all_variants, total_variants, failed = self._run_parallel(
                original_data, enhance_config, max_workers
            )

        self.logger.info(f"✅ 增强完成: 原始 {len(all_original)}, 变体 {total_variants}")
        if failed:
            self.logger.warning(f"失败对话: {len(failed)}")

        # 保存结果
        all_dialogues = all_original + all_variants
        combined_json = output_dir / f"combined_augmented_{timestamp}.json"
        combined_jsonl = output_dir / f"combined_augmented_{timestamp}.jsonl"
        variants_json = output_dir / f"variants_only_{timestamp}.json"
        variants_jsonl = output_dir / f"variants_only_{timestamp}.jsonl"

        with open(combined_json, "w", encoding="utf-8") as f:
            json.dump(all_dialogues, f, ensure_ascii=False, indent=2)
        with open(combined_jsonl, "w", encoding="utf-8") as f:
            for d in all_dialogues:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        with open(variants_json, "w", encoding="utf-8") as f:
            json.dump(all_variants, f, ensure_ascii=False, indent=2)
        with open(variants_jsonl, "w", encoding="utf-8") as f:
            for d in all_variants:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        metadata = {
            "run_id": run_id,
            "step": "augment",
            "source_file": str(input_path),
            "timestamp": timestamp,
            "tag": tag,
            "config": enhance_config,
            "statistics": {
                "original_dialogues": len(original_data),
                "generated_variants": total_variants,
                "total_dialogues": len(all_dialogues),
                "failed_dialogues": failed
            }
        }
        with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"增强完成，结果保存在: {output_dir}")
        return True

    # ========== 串行模式 ==========
    def _run_serial(self, data, config):
        all_original = []
        all_variants = []
        total_variants = 0
        failed = []
        rng = random.Random(config["seed"])

        composite = CompositeAugmenter(config["composite_config"])

        for idx, dialogue in enumerate(data):
            all_original.append(dialogue)
            try:
                variants = self._enhance_dialogue(dialogue, config, rng, composite)
                all_variants.extend(variants)
                total_variants += len(variants)
            except Exception as e:
                self.logger.error(f"对话 {idx} 增强失败: {e}")
                failed.append(idx)
        return all_original, all_variants, total_variants, failed

    # ========== 并行模式 ==========
    def _run_parallel(self, data, config, max_workers):
        chunk_size = max(1, len(data) // max_workers)
        chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

        tasks = []
        for worker_id, chunk in enumerate(chunks):
            worker_seed = config["seed"] + worker_id * 1000 + 1
            tasks.append({
                "chunk": chunk,
                "config": config,
                "seed": worker_seed,
                "worker_id": worker_id,
            })

        self.logger.info(f"分 {len(tasks)} 个chunk，每个约 {chunk_size} 条对话")

        all_original = []
        all_variants = []
        total_variants = 0
        failed = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self._augment_chunk_worker, task): task
                for task in tasks
            }

            with tqdm(total=len(data), desc="语义增强", unit="dialog") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        orig, vars_list, fail_idx = future.result(timeout=3600)
                        all_original.extend(orig)
                        all_variants.extend(vars_list)
                        total_variants += len(vars_list)
                        failed.extend(fail_idx)
                        pbar.update(len(task["chunk"]))
                        pbar.set_postfix({"变体": total_variants, "失败": len(failed)})
                    except Exception as e:
                        self.logger.error(f"chunk (worker {task['worker_id']}) 失败: {e}")
                        pbar.update(len(task["chunk"]))

        return all_original, all_variants, total_variants, failed

    # ========== Worker 函数 ==========
    @staticmethod
    def _augment_chunk_worker(task):
        chunk = task["chunk"]
        config = task["config"]
        seed = task["seed"]
        rng = random.Random(seed)

        from ..augmenters import CompositeAugmenter
        composite = CompositeAugmenter(config["composite_config"])

        def enhance_dialogue(dialogue, config, rng, composite):
            messages = dialogue.get("messages", [])
            if not messages:
                return []

            enhanceable = get_enhanceable_indices(messages, config["target_roles"], config["only_loss_true"])
            if not enhanceable:
                return []

            num_variants = config["num_variants_per_dialogue"]
            if config["adaptive_variants"]:
                num_variants = max(1, min(5, len(enhanceable) // 2))

            msg_prob = config.get("message_augment_prob", 1.0)

            variants = []
            for _ in range(num_variants):
                try:
                    new_dialogue = deepcopy(dialogue)
                    new_messages = new_dialogue["messages"]
                    for idx in enhanceable:
                        if rng.random() > msg_prob:
                            continue
                        original_text = new_messages[idx].get("content", "")
                        if not original_text:
                            continue
                        new_text = composite.apply(original_text)
                        if new_text != original_text:
                            new_messages[idx]["content"] = new_text
                    variants.append(new_dialogue)
                except Exception:
                    continue
            return variants

        def get_enhanceable_indices(messages, target_roles, only_loss_true):
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

        original_list = []
        variants_list = []
        failed_indices = []

        for idx, dialogue in enumerate(chunk):
            original_list.append(dialogue)
            try:
                vars_out = enhance_dialogue(dialogue, config, rng, composite)
                variants_list.extend(vars_out)
            except Exception:
                failed_indices.append(idx)

        return original_list, variants_list, failed_indices

    # ========== 辅助方法 ==========
    def _get_latest_final_dir(self, final_root):
        if not final_root.exists():
            return None
        dirs = [d for d in final_root.iterdir() if d.is_dir() and d.name.endswith("_final")]
        if not dirs:
            return None
        dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return dirs[0].name

    def _enhance_dialogue(self, dialogue, config, rng, composite):
        messages = dialogue.get("messages", [])
        if not messages:
            return []

        enhanceable = self._get_enhanceable_indices(messages, config["target_roles"], config["only_loss_true"])
        if not enhanceable:
            return []

        num_variants = config["num_variants_per_dialogue"]
        if config["adaptive_variants"]:
            num_variants = max(1, min(5, len(enhanceable) // 2))

        msg_prob = config.get("message_augment_prob", 1.0)

        variants = []
        for _ in range(num_variants):
            try:
                new_dialogue = deepcopy(dialogue)
                new_messages = new_dialogue["messages"]
                for idx in enhanceable:
                    if rng.random() > msg_prob:
                        continue
                    original_text = new_messages[idx].get("content", "")
                    if not original_text:
                        continue
                    new_text = composite.apply(original_text)
                    if new_text != original_text:
                        new_messages[idx]["content"] = new_text
                variants.append(new_dialogue)
            except Exception:
                continue
        return variants

    def _get_enhanceable_indices(self, messages, target_roles, only_loss_true):
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