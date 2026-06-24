"""
绘图工具（基于 matplotlib）
提供轮次分布对比图绘制，若 matplotlib 不可用则跳过并打印警告
"""

import warnings
from pathlib import Path
from typing import Dict, Optional, List
import os

try:
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


def plot_turn_distribution(
    bucket_name: str,
    input_dist: Dict[int, int],
    output_dist: Dict[int, int],
    output_dir: Path,
    selected_turns: Optional[List[int]] = None,
    title_prefix: str = "",
):
    """
    绘制清洗前后轮次分布对比柱状图
    """
    if not HAS_MATPLOTLIB:
        warnings.warn("matplotlib 未安装，跳过绘图")
        return

    # 设置中文字体（尝试多个常见路径）
    try:
        font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/System/Library/Fonts/PingFang.ttc",  # macOS
            "C:/Windows/Fonts/msyh.ttc",  # Windows
            "C:/Windows/Fonts/simsun.ttc",
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                from matplotlib.font_manager import FontProperties

                plt.rcParams["font.family"] = FontProperties(fname=fp).get_name()
                break
        else:
            # 降级方案
            plt.rcParams["font.sans-serif"] = [
                "SimHei",
                "DejaVu Sans",
                "WenQuanYi Zen Hei",
            ]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    if not input_dist and not output_dist:
        return

    if selected_turns is not None:
        all_turns = sorted(selected_turns)
    else:
        all_turns = sorted(set(input_dist.keys()) | set(output_dist.keys()))

    if not all_turns:
        return

    input_counts = [input_dist.get(t, 0) for t in all_turns]
    output_counts = [output_dist.get(t, 0) for t in all_turns]

    plt.figure(figsize=(12, 6))
    x = range(len(all_turns))
    width = 0.35
    plt.bar(x, input_counts, width, label="清洗前", color="steelblue")
    plt.bar(
        [i + width for i in x], output_counts, width, label="清洗后", color="salmon"
    )
    plt.xlabel("轮次 (turn)")
    plt.ylabel("样本数量")
    title = f"{title_prefix}{bucket_name} 清洗前后轮次分布对比"
    plt.title(title)
    plt.xticks([i + width / 2 for i in x], all_turns, rotation=45)
    plt.legend()
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{bucket_name}_turn_distribution.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()


__all__ = ["plot_turn_distribution"]
