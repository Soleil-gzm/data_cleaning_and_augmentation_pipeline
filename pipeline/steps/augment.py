"""
05_augment：语义增强（增强器架构 V2）
====================================

核心特性：
1. 按 category 管理增强器（lexical / order / model）
2. 每个增强方法有独立 enabled 开关 + weight 比例 + 专属参数
3. single / multi_step 两种组合策略，均带空变体 fallback
4. 支持 enabled_categories 限制启用的类别（未启用的类别不会加载模型）
5. model 类增强器支持主进程预加载后通过 multiprocessing.Manager
   以共享字典形式注入到每个 worker 进程，避免重复加载
6. 旧配置自动迁移（augment_weights → augmenters 详细格式）
"""

import json
import random
import re
import logging
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
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
    """返回 (变体列表, 每个变体对应的已启用增强方法名列表)"""
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


# ======================================================================
# 模块级 Worker
# ======================================================================
def _augment_chunk_worker(
    task: Dict[str, Any],
) -> Tuple[List[Dict], List[Dict], List[int], List[List[str]]]:
    """
    Worker 入口：
    - shared_resources 为 Manager.dict()，主进程预加载的模型以 "name" -> bytes / 字典形式注入
    - word_to_vec 通过 task dict 直接传递（避免 Manager.dict 序列化 numpy 数组的问题）
    - worker 按 composite_config 构建 CompositeAugmenter
    """
    chunk = task["chunk"]
    config = task["config"]
    seed = task["seed"]
    shared_resources = task.get("shared_resources") or {}
    word_to_vec = task.get("word_to_vec") or {}
    rng = random.Random(seed)

    composite = CompositeAugmenter(config["composite_config"])

    if shared_resources:
        for aug in composite.augmenters:
            try:
                aug.config["shared_resources"] = shared_resources
            except Exception:
                pass

    # 将预编码向量注入到 AsrNoiseAugmenter 的 config
    if word_to_vec:
        for aug in composite.augmenters:
            if type(aug).__name__ == "AsrNoiseAugmenter":
                try:
                    aug.config["asr_noise.word_to_vec"] = word_to_vec
                except Exception:
                    pass

    for aug in composite.augmenters:
        try:
            aug.initialize()
        except Exception:
            pass

    original_list = []
    variants_list = []
    failed_indices = []
    meta_list = []

    for idx, dialogue in enumerate(chunk):
        original_list.append(dialogue)
        try:
            vars_out, metas = _enhance_dialogue(dialogue, config, rng, composite)
            variants_list.extend(vars_out)
            meta_list.extend(metas)
        except Exception:
            failed_indices.append(idx)

    return original_list, variants_list, failed_indices, meta_list


# ======================================================================
# Pipeline 步骤
# ======================================================================
class AugmentStep(PipelineStep):
    def run(self) -> bool:
        cfg = self.context.get_step_config("05_augment")

        # ---------- 输入 ----------
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

        # ---------- 输出 ----------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = cfg.get("tag", "augment")
        run_id = f"{timestamp}_augment_{tag}"

        output_base = cfg.get("output_dir") or (
            self.context.task_dir / "output_augmented_data"
        )
        output_dir = Path(output_base) / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        self._output_paths = [
            output_dir / f"combined_augmented_{timestamp}.json",
            output_dir / f"combined_augmented_{timestamp}.jsonl",
            output_dir / f"variants_only_{timestamp}.json",
            output_dir / f"variants_only_{timestamp}.jsonl",
            output_dir / "run_metadata.json",
        ]
        self._input_paths = [input_path]

        # ---------- 并行 ----------
        global_workers = self.context.config.get("executor", {}).get("max_workers", 1)
        max_workers = cfg.get("max_workers", global_workers)

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

        # 计算实际启用的增强器（enabled=true 且 weight>0）
        active_augmenters = []
        for name, sub in augmenters_cfg.items():
            if isinstance(sub, dict) and sub.get("enabled", False) and float(sub.get("weight", 1.0)) > 0:
                active_augmenters.append(name)

        self.logger.info(f"增强 run_id: {run_id}")
        self.logger.info(f"输入: {input_path}")
        self.logger.info(f"输出: {output_dir}")
        self.logger.info(f"启用增强器: {active_augmenters or '无'}")
        self.logger.info(f"启用类别: {enabled_categories or '全部'}")
        self.logger.info(
            f"组合策略: {strategy}, min_steps={min_steps}, max_steps={max_steps}"
        )
        self.logger.info(
            f"模型增强器: {'启用' if model_enabled else '未启用（将不加载模型）'}"
        )
        if max_workers > 1:
            self.logger.info(f"并行模式，进程数: {max_workers}")

        # ---------- 预加载 jieba（避免 worker 子进程重复输出初始化日志刷屏）----------
        _ensure_jieba()

        # ---------- 加载数据 ----------
        with open(input_path, "r", encoding="utf-8") as f:
            original_data = json.load(f)
        self.logger.info(f"原始对话数: {len(original_data)}")

        # ---------- 主进程预加载模型资源（共享到 worker）----------
        shared_resources: Dict[str, Any] = {}
        word_to_vec_dict: Dict[str, Any] = {}  # 单独存储，不经过 Manager.dict()
        
        if model_enabled and augmenters_cfg.get("asr_noise", {}).get("enabled", False):
            try:
                self.logger.info("主进程预加载 ASR 增强资源 ...")
                from ..augmenters.methods.model.asr_noise import AsrNoiseAugmenter

                preload_cfg = dict(augmenters_cfg["asr_noise"])
                preload_cfg["enabled"] = True
                preload_cfg["weight"] = 1.0
                preloader = AsrNoiseAugmenter(preload_cfg)
                preloader.initialize()
                self.logger.info(
                    f"  ASR 资源加载完成: abnormal_words={len(preloader.abnormal_words)}, "
                    f"prev_to_abnormals={len(preloader.prev_to_abnormals)}"
                )
                # 传递可 pickling 的资源（向量/字典）
                shared_resources["asr_noise.abnormal_words"] = preloader.abnormal_words
                shared_resources["asr_noise.abnormal_vectors"] = (
                    preloader.abnormal_vectors
                )
                shared_resources["asr_noise.word_to_idx"] = preloader.word_to_idx
                shared_resources["asr_noise.pinyin_dict"] = preloader.pinyin_dict
                shared_resources["asr_noise.prev_to_abnormals"] = (
                    preloader.prev_to_abnormals
                )
                shared_resources["asr_noise.config"] = {
                    "prob": preloader.prob,
                    "alpha": preloader.alpha,
                    "max_operations": preloader.max_operations,
                    "insert_prob": preloader.insert_prob,
                    "retry_times": preloader.retry_times,
                    "dim": preloader.dim,
                }
                
                # 预编码用户消息中的词（避免 Worker 重复加载模型）
                if preloader.encoder:
                    self.logger.info("主进程预编码用户消息中的词 ...")
                    import jieba  # 确保 jieba 已导入
                    user_words = set()
                    for d in original_data:
                        for msg in d.get("messages", []):
                            if msg.get("role") in target_roles:
                                content = msg.get("content", "")
                                if content:
                                    words = jieba.lcut(content)
                                    for w in words:
                                        w = w.strip()
                                        if w:
                                            user_words.add(w)
                    
                    if user_words:
                        self.logger.info(f"  待编码词数: {len(user_words)}")
                        word_list = list(user_words)
                        word_vecs = preloader.encoder.encode(
                            word_list, show_progress_bar=False
                        )
                        # 存储到单独 dict，通过 task 传递（避免 Manager.dict 序列化问题）
                        word_to_vec_dict = dict(zip(word_list, word_vecs))
                        self.logger.info(f"  预编码完成: {len(word_list)} 个词")
            except Exception as e:
                self.logger.error(f"ASR 资源预加载失败: {e}，将在 worker 中按需加载")

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

        # ---------- 执行 ----------
        if max_workers <= 1:
            all_original, all_variants, total_variants, failed, _ = self._run_serial(
                original_data, enhance_config, shared_resources
            )
        else:
            all_original, all_variants, total_variants, failed, _ = self._run_parallel(
                original_data, enhance_config, max_workers, shared_resources, word_to_vec_dict
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

    # ========== 串行模式 ==========
    def _run_serial(self, data, config, shared_resources):
        all_original = []
        all_variants = []
        total_variants = 0
        failed = []
        rng = random.Random(config["seed"])

        composite = CompositeAugmenter(config["composite_config"])
        if shared_resources:
            for aug in composite.augmenters:
                try:
                    aug.config["shared_resources"] = shared_resources
                except Exception:
                    pass

        for idx, dialogue in enumerate(data):
            all_original.append(dialogue)
            try:
                variants, _ = _enhance_dialogue(dialogue, config, rng, composite)
                all_variants.extend(variants)
                total_variants += len(variants)
            except Exception as e:
                self.logger.error(f"对话 {idx} 增强失败: {e}")
                failed.append(idx)
        return all_original, all_variants, total_variants, failed, None

    # ========== 并行模式（Manager 共享资源）==========
    def _run_parallel(self, data, config, max_workers, shared_resources, word_to_vec_dict=None):
        chunk_size = max(1, (len(data) + max_workers - 1) // max_workers)
        chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

        tasks = []
        for worker_id, chunk in enumerate(chunks):
            worker_seed = config["seed"] + worker_id * 1000 + 1
            tasks.append(
                {
                    "chunk": chunk,
                    "config": config,
                    "seed": worker_seed,
                    "worker_id": worker_id,
                    "word_to_vec": word_to_vec_dict or {},  # 通过 task 传递
                }
            )

        self.logger.info(f"分 {len(tasks)} 个 chunk，每个约 {chunk_size} 条对话")

        all_original = []
        all_variants = []
        total_variants = 0
        failed = []

        with Manager() as manager:
            shared = manager.dict()
            for k, v in (shared_resources or {}).items():
                try:
                    shared[k] = v
                except Exception as e:
                    self.logger.warning(f"共享资源 {k} 注入失败: {e}")

            for task in tasks:
                task["shared_resources"] = shared

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {
                    executor.submit(_augment_chunk_worker, task): task for task in tasks
                }

                with tqdm(total=len(data), desc="语义增强", unit="dialog") as pbar:
                    for future in as_completed(future_to_task):
                        task = future_to_task[future]
                        try:
                            orig, vars_list, fail_idx, _meta = future.result(
                                timeout=3600
                            )
                            all_original.extend(orig)
                            all_variants.extend(vars_list)
                            total_variants += len(vars_list)
                            failed.extend(fail_idx)
                            pbar.update(len(task["chunk"]))
                            pbar.set_postfix(
                                {"变体": total_variants, "失败": len(failed)}
                            )
                        except Exception as e:
                            self.logger.error(
                                f"chunk (worker {task['worker_id']}) 失败: {e}"
                            )
                            pbar.update(len(task["chunk"]))

        return all_original, all_variants, total_variants, failed, None

    # ========== 辅助 ==========
    def _get_latest_final_dir(self, final_root):
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
