#!/usr/bin/env python3
"""
新流水线入口
用法: python run_pipeline.py --config configs/pipeline_config_v2.yaml [--step STEP_NAME]
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import Pipeline

# 导入所有步骤、分析器、报告器以触发注册
import pipeline.steps  # noqa
import pipeline.analyzers  # noqa
import pipeline.reporters  # noqa


def main():
    parser = argparse.ArgumentParser(description="数据清洗与增强流水线 V2")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    parser.add_argument("--step", help="单独运行某个步骤", default=None)
    args = parser.parse_args()

    pipeline = Pipeline(config_path=Path(args.config))
    success = pipeline.run(step_name=args.step)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()