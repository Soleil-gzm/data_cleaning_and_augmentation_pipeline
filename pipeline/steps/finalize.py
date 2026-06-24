"""
04_finalize：应用清洗结果，标记 loss
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from ..core.step import PipelineStep


class FinalizeStep(PipelineStep):
    def run(self) -> bool:
        cfg = self.context.get_step_config("04_finalize")
        original_json = self.context.resolve_path(
            cfg.get("original_json", "{task_dir}/raw_dialogues.json")
        )
        cleaned_root = self.context.resolve_path(
            cfg.get("cleaned_root", "{task_dir}/cleaned_jsonl")
        )
        output_root = self.context.resolve_path(
            cfg.get("output_root", "{task_dir}/final_training_data")
        )
        source_run_id = cfg.get("source_run_id")

        if not original_json.exists():
            self.logger.error(f"原始对话不存在: {original_json}")
            return False

        if source_run_id:
            cleaned_dir = cleaned_root / source_run_id
        else:
            cleaned_dir = self._get_latest_clean_dir(cleaned_root)

        if cleaned_dir is None or not cleaned_dir.exists():
            self.logger.error(f"清洗结果目录不存在: {cleaned_dir}")
            return False

        run_id = cleaned_dir.name
        self.logger.info(f"使用清洗结果: {run_id}")

        with open(original_json, "r", encoding="utf-8") as f:
            dialogues = json.load(f)
        self.logger.info(f"原始对话数: {len(dialogues)}")

        kept_turns = self._collect_kept_turns(cleaned_dir)
        total_kept = sum(len(v) for v in kept_turns.values())
        self.logger.info(f"保留轮次数: {total_kept}")

        final_data = self._apply_loss(dialogues, kept_turns)

        output_dir = output_root / f"{run_id}_final"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "cleaned_training_data.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

        metadata = {
            "run_id": f"{run_id}_final",
            "source_run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "total_dialogues": len(final_data),
                "total_assistant": sum(
                    1
                    for d in final_data
                    for m in d["messages"]
                    if m.get("role") == "assistant"
                ),
                "total_loss_true": sum(
                    1
                    for d in final_data
                    for m in d["messages"]
                    if m.get("role") == "assistant" and m.get("loss") == "True"
                ),
            },
        }
        with open(output_dir / "run_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"✅ 最终数据已保存: {output_file}")
        self._output_paths = [output_file]
        return True

    def _get_latest_clean_dir(self, cleaned_root: Path):
        if not cleaned_root.exists():
            return None
        # 匹配包含 "_clean_" 或以 "_clean" 结尾的目录
        dirs = [
            d
            for d in cleaned_root.iterdir()
            if d.is_dir() and ("_clean_" in d.name or d.name.endswith("_clean"))
        ]
        if not dirs:
            return None
        dirs.sort(reverse=True)
        return dirs[0]

    def _collect_kept_turns(self, cleaned_dir: Path):
        kept = defaultdict(set)
        for bucket_dir in cleaned_dir.iterdir():
            if not bucket_dir.is_dir():
                continue
            for jsonl_file in bucket_dir.glob("*.jsonl"):
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            dialog_id = data.get("id")
                            turn = data.get("turn")
                            if dialog_id is not None and turn is not None:
                                kept[dialog_id].add(turn)
                        except json.JSONDecodeError:
                            pass
        return kept

    def _apply_loss(self, dialogues, kept_turns):
        total_assistant = 0
        total_true = 0
        for dialog_id, dialog in enumerate(dialogues):
            messages = dialog.get("messages", [])
            assistant_indices = []
            for idx, msg in enumerate(messages):
                if msg.get("role") == "assistant":
                    msg["loss"] = "False"
                    assistant_indices.append(idx)
                    total_assistant += 1
            for turn in kept_turns.get(dialog_id, set()):
                if turn < len(assistant_indices):
                    msg_idx = assistant_indices[turn]
                    messages[msg_idx]["loss"] = "True"
                    total_true += 1
        self.logger.info(f"统计: assistant={total_assistant}, True={total_true}")
        return dialogues

    def _get_output_paths(self):
        return getattr(self, "_output_paths", [])
