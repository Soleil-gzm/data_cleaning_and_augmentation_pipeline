"""
Matplotlib 绘图报告器
"""

import warnings
from pathlib import Path
from typing import Dict, Any

from .base import BaseReporter
from ..utils.plot_utils import plot_turn_distribution


class MatplotlibReporter(BaseReporter):
    def report(self, analysis_data: Dict[str, Any], output_dir: Path, step_name: str):
        output_dir.mkdir(parents=True, exist_ok=True)

        # 尝试从分析数据中提取轮次分布信息
        # 可能的结构：
        # 1. analysis_data["analyses"]["TurnDistributionAnalyzer"]["turn_distribution"]
        # 2. analysis_data["raw_metrics"]["input_turn_dist"] 和 ["output_turn_dist"]
        turn_data = None

        # 先尝试从分析结果中获取
        analyses = analysis_data.get("analyses", {})
        for ana_name, ana_result in analyses.items():
            if "turn_distribution" in ana_result:
                turn_data = ana_result["turn_distribution"]
                break
            if "turn_retention" in ana_result:
                # 如果有 turn_retention，也可以绘制
                pass

        # 如果没有分析结果，直接使用 raw_metrics
        if turn_data is None:
            raw = analysis_data.get("raw_metrics", {})
            input_dist = raw.get("input_turn_dist", {})
            output_dist = raw.get("output_turn_dist", {})
            if input_dist or output_dist:
                plot_turn_distribution(
                    bucket_name="overall",
                    input_dist=input_dist,
                    output_dist=output_dist,
                    output_dir=output_dir,
                    title_prefix=f"{step_name}_",
                )
                self.logger.info(
                    f"分布图已保存: {output_dir}/overall_turn_distribution.png"
                )
                return

        # 如果有 turn_data 并且包含 input/output 键
        if turn_data and isinstance(turn_data, dict):
            input_dist = turn_data.get("input", {})
            output_dist = turn_data.get("output", {})
            if input_dist or output_dist:
                # 同时尝试为每个桶绘制
                # 但如果数据是按桶聚合的，这里简单处理
                plot_turn_distribution(
                    bucket_name="overall",
                    input_dist=input_dist,
                    output_dist=output_dist,
                    output_dir=output_dir,
                    title_prefix=f"{step_name}_",
                )
                self.logger.info(
                    f"分布图已保存: {output_dir}/overall_turn_distribution.png"
                )
                return

        # 如果都没有，输出警告
        self.logger.warning("分析数据中无轮次分布信息，跳过绘图")
