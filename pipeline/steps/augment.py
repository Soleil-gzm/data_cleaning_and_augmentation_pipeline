"""
05_augment：语义增强

功能描述：
  - 对对话数据中的消息进行语义增强，生成变体
  - 支持多种增强方法（词法替换、语序重排、模型增强等）
  - 当前仅支持串行执行模式

增强方法类别：
  - lexical（词法替换类）：insert_filler, stutter, homophone, synonym_replace,
                           random_delete, random_entity_replace, word_repetition
  - order（语序重排类）：reorder
  - model（模型类）：asr_noise（需要预加载模型）

配置方式：
  通过 augmenters 配置项启用/禁用各增强器，并设置权重和参数
"""

import json
import random
import re
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

from tqdm import tqdm

from ..core.step import PipelineStep
from ..augmenters import CompositeAugmenter, AugmenterRegistry, categories as _cats
from ..augmenters.utils import _ensure_jieba

# 触发所有增强器的注册（import 时完成）
from ..augmenters import *  # noqa

NAME_ALIAS = {
    "similarword": "synonym_replace",
    "synonym": "synonym_replace",
    "entity_replace": "random_entity_replace",
}


def _migrate_legacy_weights(
    augment_weights: Optional[Dict[str, float]],
) -> Dict[str, Dict[str, Any]]:
    """将旧格式 augment_weights 迁移到新格式 augmenters"""
    if not augment_weights:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for name, w in augment_weights.items():
        try:
            weight = float(w)
        except Exception:
            weight = 0.0
        if weight <= 0:
            continue
        real = NAME_ALIAS.get(name, name)
        out[real] = {
            "enabled": True,
            "weight": weight,
        }
    return out


def _build_augmenters_cfg(
    augmenters_user: Dict[str, Dict[str, Any]],
    asr_cache: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """合并用户 augmenters 配置与 ASR 模型/缓存路径"""
    out: Dict[str, Dict[str, Any]] = {}
    for name, sub in (augmenters_user or {}).items():
        if not isinstance(sub, dict):
            continue
        real = NAME_ALIAS.get(name, name)
        merged = dict(sub)
        if real == "asr_noise" and asr_cache:
            for key in (
                "vectors_path",
                "pinyin_path",
                "prev_map_path",
                "model_path",
                "model_name",
            ):
                if (
                    key not in merged
                    and key in asr_cache
                    and asr_cache[key] is not None
                ):
                    merged[key] = asr_cache[key]
        out[real] = merged
    return out


def _get_enhanceable_indices(messages, target_roles, only_loss_true):
    """获取可增强消息的索引列表"""
    indices = []
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role not in target_roles:
            continue
        content = msg.get("content", "")
        if not content or not str(content).strip():
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


def _apply_text_to_content(
    content: str,
    composite: CompositeAugmenter,
    strategy: str,
    min_steps: int,
    max_steps: int,
    rng: random.Random,
) -> str:
    """对文本内容应用增强（处理斜杠分隔的多部分内容）"""
    if not isinstance(content, str) or not content.strip():
        return content

    if "/" in content or "／" in content:
        parts = re.split(r"[／/]", content)
        enhanced_parts = []
        for p in parts:
            enhanced_parts.append(
                _apply_single(p, composite, strategy, min_steps, max_steps, rng)
            )
        return "/".join(enhanced_parts)
    return _apply_single(content, composite, strategy, min_steps, max_steps, rng)


def _apply_single(
    text: str,
    composite: CompositeAugmenter,
    strategy: str,
    min_steps: int,
    max_steps: int,
    rng: random.Random,
) -> str:
    """对单段文本应用增强策略"""
    if strategy == "multi_step":
        return composite.multi_step_apply(
            text, min_steps=min_steps, max_steps=max_steps, rng=rng
        )
    return composite.apply(text, rng=rng)


def _enhance_dialogue(
    dialogue: Dict[str, Any],
    config: Dict[str, Any],
    rng: random.Random,
    composite: CompositeAugmenter,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """对单个对话进行增强，返回 (变体列表, 每个变体对应的已启用增强方法名列表)"""
    messages = dialogue.get("messages", [])
    if not messages:
        return [], []

    enhanceable = _get_enhanceable_indices(
        messages, config["target_roles"], config["only_loss_true"]
    )
    if not enhanceable:
        return [], []

    num_variants = config["num_variants_per_dialogue"]
    if config.get("adaptive_variants", False):
        num_variants = max(1, min(3, len(enhanceable)))

    msg_prob = config.get("message_augment_prob", 1.0)
    strategy = config.get("strategy", "single")
    min_steps = config.get("min_steps", 1)
    max_steps = config.get("max_steps", min_steps)

    applied_names = composite.enabled_names()

    variants = []
    variant_meta = []
    for _ in range(num_variants):
        try:
            new_dialogue = deepcopy(dialogue)
            new_messages = new_dialogue["messages"]
            changed = False
            for idx in enhanceable:
                if rng.random() > msg_prob:
                    continue
                original_text = new_messages[idx].get("content", "")
                if not original_text:
                    continue
                new_text = _apply_text_to_content(
                    original_text, composite, strategy, min_steps, max_steps, rng
                )
                if new_text != original_text:
                    new_messages[idx]["content"] = new_text
                    changed = True
            if changed:
                variants.append(new_dialogue)
                variant_meta.append(list(applied_names))
        except Exception:
            continue
    return variants, variant_meta


class AugmentStep(PipelineStep):
    def run(self) -> bool:
        cfg = self.context.get_step_config("05_augment")

        # ---------- 输入 ----------
        source_run_id = cfg.get("source_run_id")
        input_file = cfg.get("input_file")

        if input_file:
            input_path = self.context.resolve_path(input_file)
        elif source_run_id:
            final_root = self.context.resolve_path("{task_dir}/final_training_data")
            input_path = final_root / source_run_id / "cleaned_training_data.json"
        else:
            final_root = self.context.resolve_path("{task_dir}/final_training_data")
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

        # ---------- 输出 ----------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = cfg.get("tag", "augment")
        run_id = f"{timestamp}_augment_{tag}"

        output_base = self.context.resolve_path(
            cfg.get("output_dir") or "{task_dir}/output_augmented_data"
        )
        output_dir = output_base / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        self._output_paths = [
            output_dir / f"combined_augmented_{timestamp}.json",
            output_dir / f"combined_augmented_{timestamp}.jsonl",
            output_dir / f"variants_only_{timestamp}.json",
            output_dir / f"variants_only_{timestamp}.jsonl",
            output_dir / "run_metadata.json",
        ]
        self._input_paths = [input_path]

        # ---------- 读取配置 ----------
        num_variants = cfg.get("num_variants", 3)
        target_roles = cfg.get("target_roles", ["user"])
        only_loss_true = cfg.get("only_loss_true", True)
        adaptive_variants = cfg.get("adaptive_variants", False)
        seed = cfg.get("seed", 42)
        message_augment_prob = cfg.get("message_augment_prob", 1.0)

        strategy = cfg.get("strategy", "single")
        min_steps = cfg.get("min_steps", 1)
        max_steps = cfg.get("max_steps", min_steps)
        enabled_categories = cfg.get("enabled_categories", None)

        # 迁移旧格式
        augmenters_user = cfg.get("augmenters", {})
        if not augmenters_user:
            augmenters_user = _migrate_legacy_weights(cfg.get("augment_weights"))
            if augmenters_user:
                self.logger.warning(
                    "检测到旧格式 augment_weights，已自动迁移为新 augmenters 格式；"
                    "建议后续直接使用 augmenters 详细格式以支持专属参数与 asr_cache 注入。"
                )

        asr_cache = cfg.get("asr_cache", {}) or {}
        augmenters_cfg = _build_augmenters_cfg(augmenters_user, asr_cache)

        # 模型类增强器是否被启用
        model_enabled = False
        for name, sub in augmenters_cfg.items():
            if sub.get("enabled", False) and _cats.requires_model(name):
                if (
                    enabled_categories is not None
                    and _cats.CATEGORY_MODEL not in enabled_categories
                ):
                    continue
                model_enabled = True
                break

        composite_config = {
            "augmenters": augmenters_cfg,
            "strategy": strategy,
            "default_steps": max_steps,
            "enabled_categories": enabled_categories,
            "single_retry": cfg.get("single_retry", 3),
            "multi_retry": cfg.get("multi_retry", 2),
        }

        self.logger.info(f"增强 run_id: {run_id}")
        self.logger.info(f"输入: {input_path}")
        self.logger.info(f"输出: {output_dir}")
        self.logger.info(f"启用增强器: {list(augmenters_cfg.keys())}")
        self.logger.info(f"启用类别: {enabled_categories or '全部'}")
        self.logger.info(
            f"组合策略: {strategy}, min_steps={min_steps}, max_steps={max_steps}"
        )
        self.logger.info(
            f"模型增强器: {'启用' if model_enabled else '未启用（将不加载模型）'}"
        )

        # ---------- 预加载 jieba ----------
        _ensure_jieba()

        # ---------- 加载数据 ----------
        with open(input_path, "r", encoding="utf-8") as f:
            original_data = json.load(f)
        self.logger.info(f"原始对话数: {len(original_data)}")

        # 诊断：抽样检查有多少对话有可增强的消息
        if original_data:
            sample_size = min(100, len(original_data))
            sample = original_data[:sample_size]
            has_enhanceable = sum(
                1
                for d in sample
                if _get_enhanceable_indices(
                    d.get("messages", []), target_roles, only_loss_true
                )
            )
            self.logger.info(
                f"诊断: {has_enhanceable}/{sample_size} 条对话含可增强消息 "
                f"(target_roles={target_roles}, only_loss_true={only_loss_true})"
            )
            if has_enhanceable == 0:
                self.logger.warning(
                    "⚠️ 抽样检查无任何对话包含可增强消息！"
                    "请确认 target_roles 与数据中消息的 role 字段是否匹配，"
                    "以及 only_loss_true 设置是否正确（数据中是否有 loss=True 的消息）。"
                )

        enhance_config = {
            "num_variants_per_dialogue": num_variants,
            "target_roles": target_roles,
            "only_loss_true": only_loss_true,
            "adaptive_variants": adaptive_variants,
            "message_augment_prob": message_augment_prob,
            "composite_config": composite_config,
            "strategy": strategy,
            "min_steps": min_steps,
            "max_steps": max_steps,
            "seed": seed,
        }

        # ---------- 串行执行增强 ----------
        all_original, all_variants, total_variants, failed = self._run_serial(
            original_data, enhance_config
        )

        self.logger.info(
            f"✅ 增强完成: 原始 {len(all_original)}, 变体 {total_variants}"
        )
        if failed:
            self.logger.warning(f"失败对话: {len(failed)}")

        # ---------- 保存 ----------
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
                "failed_dialogues": failed,
            },
        }
        with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        self.logger.info(f"增强完成，结果保存在: {output_dir}")
        return True

    def _run_serial(self, data, config):
        """串行执行增强"""
        all_original = []
        all_variants = []
        total_variants = 0
        failed = []
        rng = random.Random(config["seed"])

        composite = CompositeAugmenter(config["composite_config"])

        for idx, dialogue in enumerate(tqdm(data, desc="语义增强", unit="dialog")):
            all_original.append(dialogue)
            try:
                variants, _ = _enhance_dialogue(dialogue, config, rng, composite)
                all_variants.extend(variants)
                total_variants += len(variants)
            except Exception as e:
                self.logger.error(f"对话 {idx} 增强失败: {e}")
                failed.append(idx)
        return all_original, all_variants, total_variants, failed

    def _get_latest_final_dir(self, final_root):
        """获取最新的 final_training_data 目录"""
        if not final_root.exists():
            return None
        dirs = [
            d for d in final_root.iterdir() if d.is_dir() and d.name.endswith("_final")
        ]
        if not dirs:
            return None
        dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return dirs[0].name

    def _get_input_paths(self):
        return getattr(self, "_input_paths", [])

    def _get_output_paths(self):
        return getattr(self, "_output_paths", [])