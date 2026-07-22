"""
06_replace_text：文本替换 + loss 类型转换
"""

import json
from pathlib import Path

from ..core.step import PipelineStep


class ReplaceTextStep(PipelineStep):
    def run(self) -> bool:
        cfg = self.context.get_step_config("06_replace_text")
        input_file = cfg.get("input_file")
        output_file = cfg.get("output_file")
        suffix = cfg.get("suffix", "_replaced")

        if input_file:
            input_path = self.context.resolve_path(input_file)
        else:
            aug_root = self.context.resolve_path("{task_dir}/output_augmented_data")
            input_path = self._find_latest_augmented_file(aug_root)

        if input_path is None or not input_path.exists():
            self.logger.error(f"找不到输入文件: {input_path}")
            return False

        if output_file:
            output_path = self.context.resolve_path(output_file)
        else:
            stem = input_path.stem
            output_path = input_path.parent / f"{stem}{suffix}.json"

        self.logger.info(f"输入: {input_path}")
        self.logger.info(f"输出: {output_path}")

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.logger.info(f"加载 {len(data)} 条对话")

        stats = {
            "replace_platform_yqg": 0,
            "replace_yqg": 0,
            "replace_platform": 0,
            "total_replacements": 0,
            "loss_true": 0,
            "loss_false": 0,
        }

        for dialogue in data:
            messages = dialogue.get("messages", [])
            self._process_messages(messages, stats)

        self.logger.info(
            f"替换统计: '洋钱罐平台'→'华夏银行': {stats['replace_platform_yqg']}"
        )
        self.logger.info(f"替换统计: '洋钱罐'→'华夏': {stats['replace_yqg']}")
        self.logger.info(f"替换统计: '平台'→'银行': {stats['replace_platform']}")
        self.logger.info(
            f"Loss: True={stats['loss_true']}, False={stats['loss_false']}"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"✅ 替换完成: {output_path}")
        self._output_paths = [output_path]
        return True

    def _find_latest_augmented_file(self, aug_root: Path):
        if not aug_root.exists():
            return None
        candidates = []
        for subdir in aug_root.iterdir():
            if not subdir.is_dir():
                continue
            for json_file in subdir.glob("combined_augmented_*.json"):
                candidates.append(json_file)
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def _process_messages(self, messages, stats):
        for msg in messages:
            content = msg.get("content")
            if content and isinstance(content, str):
                original = content
                new_content = original
                if "洋钱罐平台" in new_content:
                    stats["replace_platform_yqg"] += new_content.count("洋钱罐平台")
                    new_content = new_content.replace("洋钱罐平台", "华夏银行")
                if "洋钱罐" in new_content:
                    stats["replace_yqg"] += new_content.count("洋钱罐")
                    new_content = new_content.replace("洋钱罐", "华夏")
                if "平台" in new_content:
                    stats["replace_platform"] += new_content.count("平台")
                    new_content = new_content.replace("平台", "银行")
                if new_content != original:
                    stats["total_replacements"] += 1
                    msg["content"] = new_content

            loss_val = msg.get("loss")
            if isinstance(loss_val, bool):
                msg["loss"] = "True" if loss_val else "False"
                if msg["loss"] == "True":
                    stats["loss_true"] += 1
                else:
                    stats["loss_false"] += 1
            elif isinstance(loss_val, str):
                if loss_val.lower() == "true":
                    stats["loss_true"] += 1
                else:
                    stats["loss_false"] += 1

    def _get_output_paths(self):
        return getattr(self, "_output_paths", [])
