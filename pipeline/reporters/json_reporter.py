import json
from pathlib import Path
from .base import BaseReporter


class JsonReporter(BaseReporter):
    def report(self, analysis_data: Dict, output_dir: Path, step_name: str):
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"{step_name}_analysis.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2)
        self.logger.info(f"JSON 报告已保存: {file_path}")