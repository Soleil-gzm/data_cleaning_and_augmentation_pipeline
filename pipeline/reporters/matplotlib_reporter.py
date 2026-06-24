import warnings
from pathlib import Path
from .base import BaseReporter
from ...utils.plot_utils import plot_turn_distribution


class MatplotlibReporter(BaseReporter):
    def report(self, analysis_data: Dict, output_dir: Path, step_name: str):
        output_dir.mkdir(parents=True, exist_ok=True)
        # 检查数据中是否有 turn_distribution 字段
        if "turn_distribution" in analysis_data:
            data = analysis_data["turn_distribution"]
            # 调用绘图工具
            plot_turn_distribution(
                bucket_name="overall",
                input_dist=data.get("input", {}),
                output_dist=data.get("output", {}),
                output_dir=output_dir,
                title_prefix=f"{step_name}_"
            )
            self.logger.info(f"分布图已保存: {output_dir}/overall_turn_distribution.png")
        else:
            self.logger.warning("分析数据中无轮次分布，跳过绘图")