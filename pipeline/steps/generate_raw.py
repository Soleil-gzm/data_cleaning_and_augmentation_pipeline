"""
00_generate_raw：从 .doc/.docx 和 .txt 生成原始对话 JSON
"""

import json
import re
from pathlib import Path
from docx import Document

from ..core.step import PipelineStep


class GenerateRawStep(PipelineStep):
    def run(self) -> bool:
        cfg = self.context.get_step_config("00_generate_raw")
        # 修改：使用 resolve_path 解析所有路径
        raw_dir = self.context.resolve_path(
            cfg.get("raw_dir")
            or self.context.config.get("paths", {})
            .get("input", {})
            .get("raw_dialogues_dir")
        )
        prompt_dir = self.context.resolve_path(
            cfg.get("prompt_dir")
            or self.context.config.get("paths", {}).get("input", {}).get("prompt_dir")
        )
        output_file = self.context.resolve_path(
            cfg.get("output_file") or str(self.context.task_dir / "raw_dialogues.json")
        )

        if not raw_dir or not prompt_dir:
            self.logger.error("缺少 raw_dir 或 prompt_dir 配置")
            return False

        self.logger.info(f"原始对话目录: {raw_dir}")
        self.logger.info(f"Prompt 目录: {prompt_dir}")
        self.logger.info(f"输出文件: {output_file}")

        multi_dialogs = []
        skipped = []

        doc_files = []
        for ext in [".docx", ".doc"]:
            doc_files.extend(raw_dir.glob(f"*{ext}"))

        if not doc_files:
            self.logger.warning(f"未找到任何 .doc/.docx 文件: {raw_dir}")
            return False

        self.logger.info(f"找到 {len(doc_files)} 个对话文件")

        for doc_path in doc_files:
            filename = doc_path.name
            dialogs = self._extract_dialogs(doc_path)
            if dialogs is None:
                self.logger.warning(f"跳过: {filename} (提取失败)")
                skipped.append(filename)
                continue

            case_id = self._parse_case_id(filename)
            if case_id is None:
                self.logger.warning(f"跳过: {filename} (无法解析案例ID)")
                skipped.append(filename)
                continue

            prompt_file = prompt_dir / f"case_{case_id}.txt"
            if not prompt_file.exists():
                self.logger.warning(
                    f"跳过: {filename} (prompt 文件不存在: {prompt_file})"
                )
                skipped.append(filename)
                continue

            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt = f.read()

            system_content = (
                "# 角色\n"
                "    你是一位来自华夏银行委外阳光机构的电话专员，"
                "现在你需要向以下客户催收信用卡欠款。\n\n"
                "## 案例信息\n\n"
                "## 标签信息\n\n" + prompt
            )
            dialogs[0]["system"] = system_content
            dialogs[0] = {key: dialogs[0][key] for key in ["system", "input", "output"]}

            messages = self._reformat_dialogs(dialogs)
            multi_dialogs.append(messages)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(multi_dialogs, f, ensure_ascii=False, indent=4)

        self.logger.info(f"✅ 生成完成: {len(multi_dialogs)} 个对话")
        if skipped:
            self.logger.warning(f"跳过文件: {skipped}")

        self._output_paths = [output_file]
        return True

    def _extract_dialogs(self, doc_path: Path):
        conversation = []
        if doc_path.suffix == ".docx":
            doc = Document(doc_path)
            for para in doc.paragraphs:
                text = para.text.strip()
                single_turn = {}
                if text.startswith("客户:") or text.startswith("客户："):
                    sep = ":" if ":" in text else "："
                    single_turn["input"] = text.split(sep, 1)[1].strip()
                elif text.startswith("专员:") or text.startswith("专员："):
                    sep = ":" if ":" in text else "："
                    single_turn["output"] = text.split(sep, 1)[1].strip()
                if single_turn:
                    conversation.append(single_turn)
        else:
            with open(doc_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            for text in lines:
                single_turn = {}
                if text.startswith("客户:") or text.startswith("客户："):
                    sep = ":" if ":" in text else "："
                    single_turn["input"] = text.split(sep, 1)[1].strip()
                elif text.startswith("专员:") or text.startswith("专员："):
                    sep = ":" if ":" in text else "："
                    single_turn["output"] = text.split(sep, 1)[1].strip()
                if single_turn:
                    conversation.append(single_turn)

        if not conversation:
            return None

        if list(conversation[0].keys())[0] == "output":
            conversation.insert(0, {"input": ""})
        if list(conversation[-1].keys())[0] == "input":
            conversation.pop()

        for i in range(len(conversation)):
            keys = list(conversation[i].keys())
            if i % 2 == 0 and "input" not in keys:
                return None
            if i % 2 == 1 and "output" not in keys:
                return None

        if len(conversation) % 2 != 0:
            conversation = conversation[:-1]

        dialogs = [
            {**conversation[i], **conversation[i + 1]}
            for i in range(0, len(conversation), 2)
        ]
        return dialogs

    def _parse_case_id(self, filename: str) -> int:
        patterns = [
            r"案例(\d+)",
            r"case_(\d+)",
            r"case(\d+)",
            r"(\d+)",
        ]
        for pat in patterns:
            m = re.search(pat, filename)
            if m:
                return int(m.group(1))
        return None

    def _reformat_dialogs(self, dialogs):
        messages = []
        for i, item in enumerate(dialogs):
            if i == 0 and "system" in item:
                messages.append({"role": "system", "content": item["system"]})
            if "input" in item:
                messages.append({"role": "user", "content": item["input"]})
            if "output" in item:
                messages.append(
                    {"role": "assistant", "content": item["output"], "loss": "True"}
                )
        return {"messages": messages}

    def _get_output_paths(self):
        return getattr(self, "_output_paths", [])
