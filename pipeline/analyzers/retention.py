"""
保留率分析器
"""

from .base import BaseAnalyzer


class RetentionAnalyzer(BaseAnalyzer):
    def analyze(self, raw_metrics: dict) -> dict:
        result = {
            "total_input": raw_metrics.get("total_input", 0),
            "total_output": raw_metrics.get("total_output", 0),
            "overall_retention_rate": 0.0,
            "buckets": {},
        }
        total_in = result["total_input"]
        total_out = result["total_output"]
        if total_in > 0:
            result["overall_retention_rate"] = total_out / total_in

        for bucket_name, stats in raw_metrics.get("buckets", {}).items():
            inp = stats.get("input_samples", 0)
            out = stats.get("output_samples", 0)
            result["buckets"][bucket_name] = {
                "input": inp,
                "output": out,
                "retention_rate": out / inp if inp > 0 else 0,
            }

        return result
