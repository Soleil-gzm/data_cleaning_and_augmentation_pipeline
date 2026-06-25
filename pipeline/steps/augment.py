"""
05_augment：语义增强（并行版）
支持多进程并行处理对话，输出带 run_id 隔离
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core.step import PipelineStep
from common import augment_utils_add as aug_utils


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
                input_path = (
                    final_root / latest_final_dir / "cleaned_training_data.json"
                )
            else:
                self.logger.error(
                    "无法确定输入文件，请配置 source_run_id 或 input_file"
                )
                return False

        if not input_path.exists():
            self.logger.error(f"输入文件不存在: {input_path}")
            return False

        # 生成 run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = cfg.get("tag", "augment")
        run_id = f"{timestamp}_augment_{tag}"

        # 输出目录
        output_base = cfg.get("output_dir") or (
            self.context.task_dir / "output_augmented_data"
        )
        output_dir = Path(output_base) / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # 并行配置
        global_workers = self.context.config.get("executor", {}).get("max_workers", 1)
        max_workers = cfg.get("max_workers", global_workers)
        if max_workers <= 1:
            self.logger.info("串行模式运行增强")
        else:
            self.logger.info(f"并行模式运行增强，进程数: {max_workers}")

        # 参数
        num_variants = cfg.get("num_variants", 3)
        target_roles = cfg.get("target_roles", ["user"])
        only_loss_true = cfg.get("only_loss_true", True)
        adaptive_variants = cfg.get("adaptive_variants", False)
        seed = cfg.get("seed", 42)
        message_augment_prob = cfg.get("message_augment_prob", 1.0)
        augment_weights = cfg.get("augment_weights", None)
        if augment_weights:
            augment_weights = {k: float(v) for k, v in augment_weights.items()}

        self.logger.info(f"增强 run_id: {run_id}")
        self.logger.info(f"输入: {input_path}")
        self.logger.info(f"输出: {output_dir}")

        # 加载 ASR 增强器（若配置）
        asr_cache_cfg = cfg.get("asr_cache", {})
        if asr_cache_cfg:
            self._load_asr_augmenter(asr_cache_cfg)

        # 加载数据
        with open(input_path, "r", encoding="utf-8") as f:
            original_data = json.load(f)
        self.logger.info(f"原始对话数: {len(original_data)}")

        # 构建增强配置（传给worker）
        enhance_config = {
            "num_variants_per_dialogue": num_variants,
            "target_roles": target_roles,
            "only_loss_true": only_loss_true,
            "adaptive_variants": adaptive_variants,
            "message_augment_prob": message_augment_prob,
            "augment_kwargs": {
                "num_variants": 1,
                "min_steps": 2,
                "max_steps": 3,
                "augment_weights": augment_weights,
            },
            "seed": seed,
        }

        # 执行增强
        if max_workers <= 1:
            all_original, all_variants, total_variants, failed = self._run_serial(
                original_data, enhance_config
            )
        else:
            all_original, all_variants, total_variants, failed = self._run_parallel(
                original_data, enhance_config, max_workers
            )

        self.logger.info(
            f"✅ 增强完成: 原始 {len(all_original)}, 变体 {total_variants}"
        )
        if failed:
            self.logger.warning(
                f"失败对话: {len(failed)} (索引: {failed[:10]}{'...' if len(failed)>10 else ''})"
            )

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

        # 元数据
        metadata = {
            "run_id": run_id,
            "step": "augment",
            "source_run_id": source_run_id or input_path.parent.name,
            "source_file": str(input_path),
            "timestamp": timestamp,
            "tag": tag,
            "config": enhance_config,
            "max_workers": max_workers,
            "statistics": {
                "original_dialogues": len(original_data),
                "generated_variants": total_variants,
                "total_dialogues": len(all_dialogues),
                "failed_dialogues": failed,
            },
            "output_files": {
                "combined": [str(combined_json), str(combined_jsonl)],
                "variants_only": [str(variants_json), str(variants_jsonl)],
            },
        }
        metadata_path = output_dir / "run_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"增强完成，结果保存在: {output_dir}")
        return True

    # ================== 串行模式 ==================
    def _run_serial(self, data, config):
        """串行增强（原有逻辑）"""
        all_original = []
        all_variants = []
        total_variants = 0
        failed = []
        rng = random.Random(config["seed"])

        # 使用tqdm显示进度
        from tqdm import tqdm

        for idx in tqdm(range(len(data)), desc="语义增强", unit="dialog"):
            dialogue = data[idx]
            all_original.append(dialogue)
            try:
                variants = self._enhance_dialogue(dialogue, config, rng, idx)
                all_variants.extend(variants)
                total_variants += len(variants)
            except Exception as e:
                self.logger.error(f"对话 {idx} 增强失败: {e}")
                failed.append(idx)

        return all_original, all_variants, total_variants, failed

    # ================== 并行模式 ==================
    def _run_parallel(self, data, config, max_workers):
        """多进程并行增强"""
        # 分割数据为chunks
        chunk_size = max(1, len(data) // max_workers)
        chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

        # 准备任务参数（每个chunk需要独立的种子偏移）
        tasks = []
        for worker_id, chunk in enumerate(chunks):
            # 为每个worker设置不同种子
            worker_seed = config["seed"] + worker_id * 1000 + 1
            tasks.append(
                {
                    "chunk": chunk,
                    "config": config,
                    "seed": worker_seed,
                    "worker_id": worker_id,
                }
            )

        self.logger.info(f"分 {len(tasks)} 个chunk，每个chunk约 {chunk_size} 条对话")

        # 收集所有结果
        all_original = []
        all_variants = []
        total_variants = 0
        failed = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self._augment_chunk_worker, task): task
                for task in tasks
            }

            # 进度条：按对话总数更新
            total_dialogs = len(data)
            with tqdm(total=total_dialogs, desc="语义增强", unit="dialog") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result(timeout=3600)  # 1小时超时
                        # result: (original_list, variants_list, failed_indices)
                        orig, vars_list, fail_idx = result
                        all_original.extend(orig)
                        all_variants.extend(vars_list)
                        total_variants += len(vars_list)
                        failed.extend(fail_idx)
                        # 更新进度条
                        pbar.update(len(task["chunk"]))
                        pbar.set_postfix({"变体": total_variants, "失败": len(failed)})
                    except Exception as e:
                        self.logger.error(
                            f"chunk (worker {task['worker_id']}) 失败: {e}"
                        )
                        # 即使失败，也要更新进度条（但无法得知具体处理了多少，按chunk大小估算）
                        pbar.update(len(task["chunk"]))

        return all_original, all_variants, total_variants, failed

    # ================== Worker 静态方法 ==================
    @staticmethod
    def _augment_chunk_worker(task):
        """
        处理一个chunk（在子进程中运行）
        返回: (original_list, variants_list, failed_indices)
        """
        chunk = task["chunk"]
        config = task["config"]
        seed = task["seed"]

        # 创建独立的随机数生成器
        rng = random.Random(seed)

        # 从config中重建增强参数
        enhance_config = {
            "num_variants_per_dialogue": config["num_variants_per_dialogue"],
            "target_roles": config["target_roles"],
            "only_loss_true": config["only_loss_true"],
            "adaptive_variants": config["adaptive_variants"],
            "message_augment_prob": config["message_augment_prob"],
            "augment_kwargs": config["augment_kwargs"],
        }

        # 由于worker无法访问self，需要局部函数实现增强逻辑
        def enhance_single(dialogue, idx):
            messages = dialogue.get("messages", [])
            if not messages:
                return []

            enhanceable = AugmentStep._get_enhanceable_indices(
                messages,
                enhance_config["target_roles"],
                enhance_config["only_loss_true"],
            )
            if not enhanceable:
                return []

            num_variants = enhance_config["num_variants_per_dialogue"]
            if enhance_config["adaptive_variants"]:
                num_variants = max(1, min(5, len(enhanceable) // 2))

            variants = []
            aug_kwargs = enhance_config["augment_kwargs"]
            msg_prob = enhance_config.get("message_augment_prob", 1.0)

            for _ in range(num_variants):
                try:
                    new_dialogue = deepcopy(dialogue)
                    new_messages = new_dialogue["messages"]
                    for idx_pos in enhanceable:
                        if rng.random() > msg_prob:
                            continue
                        original_text = new_messages[idx_pos].get("content", "")
                        if not original_text:
                            continue
                        # 调用增强函数
                        variants_list = aug_utils.augment_cell_multi(
                            original_text, **aug_kwargs
                        )
                        if variants_list and variants_list[0] != original_text:
                            new_messages[idx_pos]["content"] = variants_list[0]
                    variants.append(new_dialogue)
                except Exception:
                    continue
            return variants

        original_list = []
        variants_list = []
        failed_indices = []

        for idx, dialogue in enumerate(chunk):
            original_list.append(dialogue)
            try:
                vars_out = enhance_single(dialogue, idx)
                variants_list.extend(vars_out)
            except Exception:
                failed_indices.append(idx)

        return original_list, variants_list, failed_indices

    # ================== 辅助方法 ==================
    @staticmethod
    def _get_enhanceable_indices(messages, target_roles, only_loss_true):
        """获取可增强的消息索引"""
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

    def _get_latest_final_dir(self, final_root: Path):
        """获取最新的 final_training_data 子目录"""
        if not final_root.exists():
            return None
        dirs = [
            d for d in final_root.iterdir() if d.is_dir() and d.name.endswith("_final")
        ]
        if not dirs:
            return None
        dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return dirs[0].name

    def _load_asr_augmenter(self, asr_cfg: dict):
        """加载 ASR 增强器"""
        try:
            project_root = self.context.intermediate_root.parent
            vectors_path = asr_cfg.get("vectors_path")
            pinyin_path = asr_cfg.get("pinyin_path")
            prev_map_path = asr_cfg.get("prev_map_path")
            model_path = asr_cfg.get("model_path")

            if vectors_path and not Path(vectors_path).is_absolute():
                vectors_path = project_root / vectors_path
            if pinyin_path and not Path(pinyin_path).is_absolute():
                pinyin_path = project_root / pinyin_path
            if prev_map_path and not Path(prev_map_path).is_absolute():
                prev_map_path = project_root / prev_map_path
            if model_path and not Path(model_path).is_absolute():
                model_path = project_root / model_path

            from common.asr_noise_augmenter import AsrNoiseAugmenter

            asr_augmenter = AsrNoiseAugmenter(
                vectors_path=vectors_path,
                pinyin_path=pinyin_path,
                prev_map_path=(
                    prev_map_path
                    if prev_map_path and Path(prev_map_path).exists()
                    else None
                ),
                model_path=model_path,
            )
            aug_utils.set_asr_augmenter(asr_augmenter)
            self.logger.info(f"ASR 增强器已加载: {model_path}")
        except Exception as e:
            self.logger.warning(f"ASR 增强器加载失败: {e}")

    # _enhance_dialogue 方法不再需要（已在worker中内联），但为了兼容可能保留但不使用
