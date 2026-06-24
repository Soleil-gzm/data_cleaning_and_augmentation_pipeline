"""
05_augment：语义增强，带进度条，预留策略接口
"""
import json
import random
import sys
import os
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core.step import PipelineStep
from ..utils.progress import get_progress_bar

# 导入原有的增强工具（保持兼容）
from common import augment_utils_add as aug_utils


class AugmentStep(PipelineStep):
    def run(self) -> bool:
        cfg = self.context.get_step_config("05_augment")

        # 获取输入文件
        source_run_id = cfg.get("source_run_id")
        input_file = cfg.get("input_file")
        if input_file:
            input_path = self.context.resolve_path(input_file)
        elif source_run_id:
            input_path = self.context.task_dir / "final_training_data" / source_run_id / "cleaned_training_data.json"
        else:
            # 自动查找最新的 final
            final_root = self.context.task_dir / "final_training_data"
            latest = self._get_latest_final_dir(final_root)
            if latest:
                input_path = final_root / latest / "cleaned_training_data.json"
            else:
                self.logger.error("无法确定输入文件，请配置 source_run_id 或 input_file")
                return False

        if not input_path.exists():
            self.logger.error(f"输入文件不存在: {input_path}")
            return False

        # 输出目录
        output_dir = cfg.get("output_dir")
        if output_dir:
            output_dir = self.context.resolve_path(output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tag = cfg.get("tag", self.context.task_name)
            output_dir = self.context.task_dir / "output_augmented_data" / f"{timestamp}_augment_{tag}"
        output_dir.mkdir(parents=True, exist_ok=True)

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

        self.logger.info(f"输入: {input_path}")
        self.logger.info(f"输出: {output_dir}")
        self.logger.info(f"变体数: {num_variants}, 目标角色: {target_roles}")

        # 加载 ASR 增强器
        asr_cache_cfg = cfg.get("asr_cache", {})
        if asr_cache_cfg:
            self._load_asr_augmenter(asr_cache_cfg)

        # 加载数据
        with open(input_path, "r", encoding="utf-8") as f:
            original_data = json.load(f)
        self.logger.info(f"原始对话数: {len(original_data)}")

        rng = random.Random(seed)

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
            }
        }

        all_original = []
        all_variants = []
        total_variants = 0
        failed = []

        # 进度条
        pbar = get_progress_bar(
            range(len(original_data)),
            desc="语义增强",
            unit="dialog",
            show=True
        )

        for idx in pbar:
            dialogue = original_data[idx]
            all_original.append(dialogue)

            try:
                variants = self._enhance_dialogue(dialogue, enhance_config, rng, idx)
                all_variants.extend(variants)
                total_variants += len(variants)
            except Exception as e:
                self.logger.error(f"对话 {idx} 增强失败: {e}")
                failed.append(idx)

            # 更新进度条后缀
            if idx % 50 == 0:
                pbar.set_postfix({"变体": total_variants})

        self.logger.info(f"✅ 增强完成: 原始 {len(original_data)}, 变体 {total_variants}")
        if failed:
            self.logger.warning(f"失败对话: {len(failed)}")

        # 保存
        all_dialogues = all_original + all_variants
        save_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        combined_json = output_dir / f"combined_augmented_{save_timestamp}.json"
        combined_jsonl = output_dir / f"combined_augmented_{save_timestamp}.jsonl"

        with open(combined_json, "w", encoding="utf-8") as f:
            json.dump(all_dialogues, f, ensure_ascii=False, indent=2)
        with open(combined_jsonl, "w", encoding="utf-8") as f:
            for d in all_dialogues:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        variants_json = output_dir / f"variants_only_{save_timestamp}.json"
        variants_jsonl = output_dir / f"variants_only_{save_timestamp}.jsonl"

        with open(variants_json, "w", encoding="utf-8") as f:
            json.dump(all_variants, f, ensure_ascii=False, indent=2)
        with open(variants_jsonl, "w", encoding="utf-8") as f:
            for d in all_variants:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        self.logger.info(f"合并文件: {combined_json}")
        self.logger.info(f"仅变体: {variants_json}")

        self._output_paths = [combined_json, variants_json, combined_jsonl, variants_jsonl]
        return True

    def _get_latest_final_dir(self, final_root: Path):
        if not final_root.exists():
            return None
        dirs = [d for d in final_root.iterdir() if d.is_dir() and d.name.endswith("_final")]
        if not dirs:
            return None
        dirs.sort(reverse=True)
        return dirs[0].name

    def _load_asr_augmenter(self, asr_cfg: dict):
        """加载 ASR 增强器（从原代码迁移）"""
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
                prev_map_path=prev_map_path if prev_map_path and Path(prev_map_path).exists() else None,
                model_path=model_path
            )
            aug_utils.set_asr_augmenter(asr_augmenter)
            self.logger.info(f"ASR 增强器已加载: {model_path}")
        except Exception as e:
            self.logger.warning(f"ASR 增强器加载失败: {e}")

    def _enhance_dialogue(self, dialogue, config, rng, dialog_id):
        """生成变体对话列表"""
        variants = []
        messages = dialogue.get("messages", [])
        if not messages:
            return variants

        enhanceable = self._get_enhanceable_indices(messages, config["target_roles"], config["only_loss_true"])
        if not enhanceable:
            return variants

        num_variants = config["num_variants_per_dialogue"]
        if config["adaptive_variants"]:
            num_variants = max(1, min(5, len(enhanceable) // 2))

        aug_kwargs = config["augment_kwargs"]
        msg_prob = config.get("message_augment_prob", 1.0)

        for var_id in range(num_variants):
            try:
                new_dialogue = deepcopy(dialogue)
                new_messages = new_dialogue["messages"]
                for idx in enhanceable:
                    if rng.random() > msg_prob:
                        continue
                    original_text = new_messages[idx].get("content", "")
                    if not original_text:
                        continue
                    variants_list = aug_utils.augment_cell_multi(original_text, **aug_kwargs)
                    if variants_list and variants_list[0] != original_text:
                        new_messages[idx]["content"] = variants_list[0]
                variants.append(new_dialogue)
            except Exception as e:
                self.logger.debug(f"对话 {dialog_id} 变体 {var_id} 失败: {e}")
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

    def _get_output_paths(self):
        return getattr(self, "_output_paths", [])