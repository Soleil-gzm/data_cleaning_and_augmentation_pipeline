"""
CSV 报告器
"""

import csv
from pathlib import Path
from typing import Dict, Any, List

from .base import BaseReporter


class CsvReporter(BaseReporter):
    def report(self, analysis_data: Dict[str, Any], output_dir: Path, step_name: str):
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"{step_name}_summary.csv"

        # 尝试将分析数据展平为表格
        # 这里处理常见的两种结构：
        # 1. {"analyses": {...}, "raw_metrics": {...}}
        # 2. 直接是 {"buckets": {...}, "total_input": ...}
        rows = []
        headers = set()

        def flatten_dict(d: Dict, prefix: str = "") -> Dict:
            """递归展平字典"""
            items = {}
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    items.update(flatten_dict(v, key))
                else:
                    items[key] = v
            return items

        # 如果是复杂的嵌套结构，展平最外层
        if "analyses" in analysis_data:
            for ana_name, ana_value in analysis_data["analyses"].items():
                if isinstance(ana_value, dict):
                    flat = flatten_dict(ana_value, ana_name)
                    rows.append(flat)
                else:
                    rows.append({ana_name: ana_value})
        else:
            # 直接展平整个 dict
            rows.append(flatten_dict(analysis_data))

        # 收集所有列名
        for row in rows:
            headers.update(row.keys())

        headers = sorted(headers)

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                # 确保所有键存在
                complete_row = {h: row.get(h, "") for h in headers}
                writer.writerow(complete_row)

        self.logger.info(f"CSV 报告已保存: {file_path}")
