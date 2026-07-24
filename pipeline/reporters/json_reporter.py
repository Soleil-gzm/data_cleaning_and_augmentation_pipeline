"""
JSON 报告器
"""

from pathlib import Path
from typing import Dict, Any

from .base import BaseReporter
from ..io import write_json


class JsonReporter(BaseReporter):
    def report(self, analysis_data: Dict[str, Any], output_dir: Path, step_name: str):
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"{step_name}_analysis.json"
        write_json(analysis_data, file_path)
        self.logger.info(f"JSON 报告已保存: {file_path}")
