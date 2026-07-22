"""
02_bucket：按 turn 分桶

当前实现：
  - 仅支持 manual（手动）分桶策略
  - 通过配置项 manual_buckets 定义桶的边界

未来扩展预留：
  - strategy 参数保留，便于将来添加其他策略（如 percentile、equal_count）
  - _get_bucket_name 和 _get_bucket_for_turn 方法保持通用设计
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict

from ..core.step import PipelineStep


class BucketStep(PipelineStep):
    def run(self) -> bool:
        cfg = self.context.get_step_config("02_bucket")
        samples_dir = self.context.resolve_path(
            cfg.get("samples_dir", "{task_dir}/samples")
        )
        output_base = self.context.resolve_path(
            cfg.get("output_base", "{task_dir}/bucketed")
        )
        strategy = cfg.get("strategy", "manual")
        manual_buckets = cfg.get("manual_buckets", [])

        if not samples_dir.exists():
            self.logger.error(f"样本目录不存在: {samples_dir}")
            return False

        if output_base.exists():
            self.logger.info(f"清空旧桶目录: {output_base}")
            shutil.rmtree(output_base)

        output_base.mkdir(parents=True, exist_ok=True)

        if strategy == "manual":
            buckets = [(low, high) for low, high in manual_buckets]
        else:
            self.logger.error(f"未知策略: {strategy}")
            return False

        if not buckets:
            self.logger.error("未配置任何桶，请检查 manual_buckets 配置")
            return False

        self.logger.info(f"分桶策略: {strategy}, 共 {len(buckets)} 个桶")

        bucket_dirs = {}
        for idx, (low, high) in enumerate(buckets):
            bucket_name = self._get_bucket_name(idx, low, high)
            bucket_dir = output_base / bucket_name
            bucket_dir.mkdir(exist_ok=True)
            bucket_dirs[bucket_name] = bucket_dir

        jsonl_files = list(samples_dir.glob("*.jsonl"))
        if not jsonl_files:
            self.logger.warning(f"未找到 JSONL 文件: {samples_dir}")
            return True

        self.logger.info(f"找到 {len(jsonl_files)} 个样本文件")

        total_samples = 0
        for input_file in jsonl_files:
            file_handles = {}
            try:
                with open(input_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        turn = data.get("turn")
                        if turn is None:
                            continue
                        _, bucket_name = self._get_bucket_for_turn(turn, buckets)
                        if bucket_name is None:
                            continue
                        bucket_dir = bucket_dirs[bucket_name]
                        out_file = bucket_dir / input_file.name
                        if out_file not in file_handles:
                            file_handles[out_file] = open(
                                out_file, "a", encoding="utf-8"
                            )
                        file_handles[out_file].write(line + "\n")
                        total_samples += 1
            finally:
                for h in file_handles.values():
                    h.close()

        self.logger.info(f"✅ 分桶完成，共处理 {total_samples} 条样本")
        for name, d in bucket_dirs.items():
            cnt = sum(1 for _ in d.glob("*.jsonl"))
            self.logger.info(f"  {name}: {cnt} 个文件")

        self._output_paths = [output_base]
        return True

    def _get_bucket_name(self, idx, low, high):
        if high >= 9999:
            return f"bucket_{low}_plus"
        elif low == high:
            return f"bucket_{low}"
        else:
            return f"bucket_{low}_{high}"

    def _get_bucket_for_turn(self, turn, buckets):
        for idx, (low, high) in enumerate(buckets):
            if high >= 9999:
                if turn >= low:
                    return idx, self._get_bucket_name(idx, low, high)
            elif low <= turn <= high:
                return idx, self._get_bucket_name(idx, low, high)
        return None, None

    def _get_output_paths(self):
        return getattr(self, "_output_paths", [])