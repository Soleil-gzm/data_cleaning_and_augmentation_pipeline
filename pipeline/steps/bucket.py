"""
02_bucket：按 turn 分桶
"""
import json
from pathlib import Path
from collections import defaultdict

from ..core.step import PipelineStep


class BucketStep(PipelineStep):
    def run(self) -> bool:
        cfg = self.context.get_step_config("02_bucket")
        samples_dir = cfg.get("samples_dir") or (self.context.task_dir / "samples")
        output_base = cfg.get("output_base") or (self.context.task_dir / "bucketed")
        strategy = cfg.get("strategy", "percentile")
        auto_params = cfg.get("auto_params", {"percentiles": [0, 25, 50, 75, 90, 95, 100]})
        manual_buckets = cfg.get("manual_buckets", [])

        samples_dir = Path(samples_dir)
        output_base = Path(output_base)

        if not samples_dir.exists():
            self.logger.error(f"样本目录不存在: {samples_dir}")
            return False

        # 清空旧桶
        if output_base.exists():
            import shutil
            self.logger.info(f"清空旧桶目录: {output_base}")
            shutil.rmtree(output_base)

        output_base.mkdir(parents=True, exist_ok=True)

        # 加载轮次分布（自动模式需要）
        if strategy in ["auto", "percentile", "equal_count"]:
            stats_path = self.context.task_dir / "stats" / "turn_distribution.json"
            if not stats_path.exists():
                self.logger.error(f"统计文件不存在: {stats_path}，请先运行 01_split")
                return False
            with open(stats_path, "r") as f:
                stats = json.load(f)
            turn_dist = {int(k): v for k, v in stats.get("turn_distribution", {}).items()}
            self.logger.info(f"加载轮次分布，共 {len(turn_dist)} 种轮次")
        else:
            turn_dist = None

        # 生成桶边界
        if strategy in ["auto", "percentile"]:
            actual_strategy = "percentile"
            buckets = self._auto_buckets(turn_dist, actual_strategy, auto_params)
        elif strategy == "equal_count":
            buckets = self._auto_buckets(turn_dist, "equal_count", auto_params)
        elif strategy == "manual":
            buckets = [(low, high) for low, high in manual_buckets]
        else:
            self.logger.error(f"未知策略: {strategy}")
            return False

        self.logger.info(f"分桶策略: {strategy}, 共 {len(buckets)} 个桶")

        # 创建桶目录
        bucket_dirs = {}
        for idx, (low, high) in enumerate(buckets):
            bucket_name = self._get_bucket_name(idx, low, high)
            bucket_dir = output_base / bucket_name
            bucket_dir.mkdir(exist_ok=True)
            bucket_dirs[bucket_name] = bucket_dir

        # 处理所有 JSONL 文件
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
                            file_handles[out_file] = open(out_file, "a", encoding="utf-8")
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

    def _auto_buckets(self, turn_dist, strategy, params):
        turns = sorted(turn_dist.keys())
        counts = [turn_dist[t] for t in turns]
        total = sum(counts)

        if strategy == "percentile":
            percentiles = params.get("percentiles", [0, 25, 50, 75, 90, 95, 100])
            cumulative = []
            running = 0
            for cnt in counts:
                running += cnt
                cumulative.append(running / total)
            boundaries = set()
            for p in percentiles:
                target = p / 100.0
                for i, cum in enumerate(cumulative):
                    if cum >= target:
                        boundaries.add(turns[i])
                        break
            boundaries = sorted(boundaries)
            buckets = []
            for i in range(len(boundaries) - 1):
                low = boundaries[i]
                high = boundaries[i+1] - 1 if boundaries[i+1] > boundaries[i] else boundaries[i]
                if low <= high:
                    buckets.append((low, high))
            buckets.append((boundaries[-1], float("inf")))
            return buckets

        elif strategy == "equal_count":
            min_bucket_size = params.get("min_bucket_size", 1000)
            buckets = []
            start = turns[0]
            cum_cnt = 0
            for t, cnt in zip(turns, counts):
                cum_cnt += cnt
                if cum_cnt >= min_bucket_size:
                    buckets.append((start, t))
                    start = t + 1
                    cum_cnt = 0
            if start <= turns[-1]:
                buckets.append((start, float("inf")))
            return buckets

        return []

    def _get_bucket_name(self, idx, low, high):
        if high == float("inf"):
            return f"bucket_{low}_plus"
        elif low == high:
            return f"bucket_{low}"
        else:
            return f"bucket_{low}_{high}"

    def _get_bucket_for_turn(self, turn, buckets):
        for idx, (low, high) in enumerate(buckets):
            if low <= turn <= high:
                return idx, self._get_bucket_name(idx, low, high)
        return None, None

    def _get_output_paths(self):
        return getattr(self, "_output_paths", [])