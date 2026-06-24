"""
01_split：多轮对话拆分为样本，带进度条
"""
import json
from pathlib import Path
from collections import defaultdict

from ..core.step import PipelineStep
from ..utils.progress import get_progress_bar


class SplitDialoguesStep(PipelineStep):
    def run(self) -> bool:
        cfg = self.context.get_step_config("01_split")
        input_json = cfg.get("input_json") or (self.context.task_dir / "raw_dialogues.json")
        output_dir = cfg.get("output_dir") or (self.context.task_dir / "samples")
        stats_dir = cfg.get("stats_dir") or (self.context.task_dir / "stats")
        batch_size = cfg.get("batch_size", 120000)

        input_path = Path(input_json)
        if not input_path.exists():
            self.logger.error(f"输入文件不存在: {input_path}")
            return False

        output_dir = Path(output_dir)
        stats_dir = Path(stats_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stats_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"输入: {input_path}")
        self.logger.info(f"输出目录: {output_dir}")

        # 加载原始数据
        with open(input_path, "r", encoding="utf-8") as f:
            dialogues = json.load(f)

        total = len(dialogues)
        self.logger.info(f"总对话数: {total}")

        turn_counter = defaultdict(int)
        batch_start = 0
        current_file = None
        current_file_path = None
        file_count = 0

        # 进度条
        pbar = get_progress_bar(range(total), desc="拆分对话", unit="dialog", show=True)

        for dialog_idx in pbar:
            dialog = dialogues[dialog_idx]
            messages = dialog.get("messages", [])
            if not messages:
                continue

            samples = self._process_dialog(dialog_idx, messages, turn_counter)

            # 写入批次文件
            if current_file is None or file_count >= batch_size:
                if current_file:
                    current_file.close()
                batch_start = dialog_idx
                batch_end = batch_start + batch_size - 1
                fname = f"sample_{batch_start:08d}_{batch_end:08d}.jsonl"
                current_file_path = output_dir / fname
                current_file = open(current_file_path, "w", encoding="utf-8")
                file_count = 0
                self.logger.debug(f"创建批次: {fname}")

            for sample in samples:
                current_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                file_count += 1

        if current_file:
            current_file.close()

        # 保存统计
        stats = {
            "total_samples": sum(turn_counter.values()),
            "turn_distribution": dict(turn_counter),
        }
        stats_path = stats_dir / "turn_distribution.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        self.logger.info(f"✅ 总样本数: {stats['total_samples']}")
        self.logger.info(f"统计已保存: {stats_path}")

        self._output_paths = [output_dir, stats_path]
        return True

    def _process_dialog(self, dialog_id: int, messages: list, turn_counter: dict):
        """处理单个对话，返回样本列表"""
        samples = []
        history_pairs = []
        pending_user = None

        for i, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                pending_user = msg
            elif role == "assistant" and pending_user is not None:
                turn = len(samples)
                user_raw = pending_user.get("content", "")
                assistant_raw = msg.get("content", "")

                history_text = ""
                for hu, ha in history_pairs:
                    history_text += f"{hu}\n{ha}\n"

                if history_text:
                    full_input = f"Q：{history_text}{user_raw}"
                else:
                    full_input = f"Q：{user_raw}" if user_raw else "Q："

                target_output = f"A：{assistant_raw}" if assistant_raw else "A："
                full_text = history_text + f"{user_raw}\n{assistant_raw}"

                sample = {
                    "id": dialog_id,
                    "turn": turn,
                    "user_input": full_input,
                    "target_output": target_output,
                    "loss": msg.get("loss", False),
                    "text": full_text,
                }
                samples.append(sample)
                turn_counter[turn] += 1
                history_pairs.append((user_raw, assistant_raw))
                pending_user = None

        return samples

    def _get_output_paths(self):
        return getattr(self, "_output_paths", [])