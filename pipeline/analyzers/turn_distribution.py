"""
轮次分布分析器
"""
from .base import BaseAnalyzer


class TurnDistributionAnalyzer(BaseAnalyzer):
    def analyze(self, raw_metrics: dict) -> dict:
        return {
            "turn_distribution": {
                "input": raw_metrics.get("input_turn_dist", {}),
                "output": raw_metrics.get("output_turn_dist", {}),
            },
            "turn_retention": self._compute_turn_retention(
                raw_metrics.get("input_turn_dist", {}),
                raw_metrics.get("output_turn_dist", {})
            )
        }

    def _compute_turn_retention(self, input_dist: dict, output_dist: dict) -> dict:
        result = {}
        all_turns = set(input_dist.keys()) | set(output_dist.keys())
        for turn in all_turns:
            inp = input_dist.get(turn, 0)
            out = output_dist.get(turn, 0)
            result[str(turn)] = {
                "input": inp,
                "output": out,
                "retention": out / inp if inp > 0 else 0
            }
        return result