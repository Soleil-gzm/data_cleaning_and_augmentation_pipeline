"""
03_clean 步骤：仅负责调用 Data-Juicer 清洗，产出原始指标
"""
import json
import subprocess
import shutil
import re
from pathlib import Path
from collections import defaultdict

from ..core.step import PipelineStep
from ..utils.progress import get_progress_bar
from ..utils.subprocess_utils import run_subprocess
from ..analyzers.registry import AnalyzerRegistry
from ..analyzers.retention import RetentionAnalyzer
from ..analyzers.turn_distribution import TurnDistributionAnalyzer
from ..reporters.registry import ReporterRegistry


class CleanStep(PipelineStep):
    def run(self) -> bool:
        cfg = self.context.get_step_config("03_clean")
        bucketed_root = cfg.get("bucketed_root") or (self.context.task_dir / "bucketed")
        cleaned_root = cfg.get("cleaned_root") or (self.context.task_dir / "cleaned_jsonl")
        trace_root = cfg.get("trace_root") or (self.context.task_dir / "trace_output")
        configs_dir = Path(cfg.get("configs_dir", "configs/configs_qa"))
        bucket_config_map = cfg.get("bucket_config_map", [])

        # 创建输出目录
        run_id = f"{self.context.task_name}_clean"
        cleaned_base = Path(cleaned_root) / run_id
        trace_base = Path(trace_root) / run_id
        cleaned_base.mkdir(parents=True, exist_ok=True)
        trace_base.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"清洗输出目录: {cleaned_base}")

        # 收集所有桶
        bucketed_root_path = Path(bucketed_root)
        if not bucketed_root_path.exists():
            self.logger.error(f"分桶目录不存在: {bucketed_root_path}")
            return False

        bucket_dirs = [d for d in bucketed_root_path.iterdir() if d.is_dir()]
        # 进度条外层
        bucket_iter = get_progress_bar(bucket_dirs, desc="清洗桶", unit="bucket", show=True)

        # 原始指标收集
        raw_metrics = {
            "buckets": {},
            "total_input": 0,
            "total_output": 0,
            "input_turn_dist": defaultdict(int),
            "output_turn_dist": defaultdict(int),
        }

        dj_process = shutil.which("dj-process")
        if dj_process is None:
            dj_process = [shutil.which("python"), "-m", "data_juicer.core.process_data"]
        else:
            dj_process = [dj_process]

        for bucket_dir in bucket_iter:
            bucket_name = bucket_dir.name
            # 匹配配置文件
            config_filename = self._get_config_for_bucket(bucket_name, bucket_config_map)
            if not config_filename:
                self.logger.warning(f"桶 {bucket_name} 未匹配配置文件，跳过")
                continue
            config_file = configs_dir / config_filename
            if not config_file.exists():
                self.logger.warning(f"配置文件 {config_file} 不存在，跳过")
                continue

            self.logger.info(f"  处理桶: {bucket_name} (配置: {config_filename})")
            output_dir = cleaned_base / bucket_name
            trace_dir = trace_base / bucket_name
            output_dir.mkdir(parents=True, exist_ok=True)
            trace_dir.mkdir(parents=True, exist_ok=True)

            input_files = list(bucket_dir.glob("*.jsonl"))
            if not input_files:
                self.logger.warning(f"桶 {bucket_name} 无 JSONL 文件")
                continue

            bucket_stats = {
                "input_samples": 0,
                "output_samples": 0,
                "input_turn_dist": defaultdict(int),
                "output_turn_dist": defaultdict(int),
            }

            # 内层文件处理（无进度条，用日志）
            for input_file in input_files:
                output_file = output_dir / input_file.name
                trace_subdir = trace_dir / input_file.stem

                input_cnt = self._count_lines(input_file)
                input_dist = self._collect_turn_dist(input_file)

                # 生成临时配置
                with open(config_file, "r") as f:
                    config_content = f.read()
                config_content = config_content.replace("__INPUT_FILE__", str(input_file.absolute()))
                config_content = config_content.replace("__OUTPUT_FILE__", str(output_file.absolute()))
                config_content = config_content.replace("work_dir:", f"work_dir: {trace_subdir}\n")

                temp_config = Path(f"temp_{input_file.stem}.yaml")
                with open(temp_config, "w") as f:
                    f.write(config_content)

                cmd = dj_process + ["--config", str(temp_config)]
                result = run_subprocess(cmd, logger=self.logger)

                temp_config.unlink()

                if result.returncode == 0 and output_file.exists():
                    output_cnt = self._count_lines(output_file)
                    output_dist = self._collect_turn_dist(output_file)
                    self.logger.info(f"      ✅ {input_file.name}: {input_cnt} → {output_cnt}")
                else:
                    output_cnt = 0
                    output_dist = defaultdict(int)
                    self.logger.warning(f"      ⚠️ {input_file.name} 清洗失败或输出为空")

                bucket_stats["input_samples"] += input_cnt
                bucket_stats["output_samples"] += output_cnt
                for t, c in input_dist.items():
                    bucket_stats["input_turn_dist"][t] += c
                    raw_metrics["input_turn_dist"][t] += c
                for t, c in output_dist.items():
                    bucket_stats["output_turn_dist"][t] += c
                    raw_metrics["output_turn_dist"][t] += c

            raw_metrics["buckets"][bucket_name] = {
                "input_samples": bucket_stats["input_samples"],
                "output_samples": bucket_stats["output_samples"],
                "input_turn_dist": dict(bucket_stats["input_turn_dist"]),
                "output_turn_dist": dict(bucket_stats["output_turn_dist"]),
                "retention_rate": bucket_stats["output_samples"] / bucket_stats["input_samples"] if bucket_stats["input_samples"] > 0 else 0
            }
            raw_metrics["total_input"] += bucket_stats["input_samples"]
            raw_metrics["total_output"] += bucket_stats["output_samples"]

        # 转换 defaultdict 为 dict
        raw_metrics["input_turn_dist"] = dict(raw_metrics["input_turn_dist"])
        raw_metrics["output_turn_dist"] = dict(raw_metrics["output_turn_dist"])

        # 保存原始指标
        metrics_path = self.context.task_dir / "reports" / run_id / "raw_clean_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(raw_metrics, f, indent=2)
        self.logger.info(f"原始清洗指标已保存: {metrics_path}")

        # ===== 触发分析器和报告器 =====
        self._run_analyzers_and_reporters(raw_metrics, run_id)

        return True

    def _get_config_for_bucket(self, bucket_name, bucket_config_map):
        for entry in bucket_config_map:
            pattern = entry.get("pattern")
            if pattern and re.match(pattern, bucket_name):
                return entry.get("config")
        return None

    def _count_lines(self, file_path):
        if not file_path.exists():
            return 0
        with open(file_path, "r") as f:
            return sum(1 for _ in f)

    def _collect_turn_dist(self, file_path):
        dist = defaultdict(int)
        if not file_path.exists():
            return dist
        with open(file_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    turn = data.get("turn")
                    if turn is not None:
                        dist[int(turn)] += 1
                except:
                    pass
        return dist

    def _run_analyzers_and_reporters(self, raw_metrics: dict, run_id: str):
        """触发配置中绑定的分析器和报告器"""
        reporting_cfg = self.context.config.get("reporting", {})
        if not reporting_cfg.get("enabled", True):
            self.logger.info("报告模块已禁用")
            return

        # 获取该步骤绑定的分析器列表
        step_cfg = self.context.get_step_config("03_clean")
        analyzer_names = step_cfg.get("attach_analyzers", ["RetentionAnalyzer", "TurnDistributionAnalyzer"])

        # 执行分析
        analysis_results = {}
        for name in analyzer_names:
            try:
                analyzer = AnalyzerRegistry.get_analyzer(name, self.context)
                self.logger.info(f"  运行分析器: {name}")
                result = analyzer.analyze(raw_metrics)
                analysis_results[name] = result
            except ValueError as e:
                self.logger.warning(f"分析器 {name} 未注册: {e}")
            except Exception as e:
                self.logger.exception(f"分析器 {name} 执行失败: {e}")

        if not analysis_results:
            return

        # 执行报告器
        reporters_cfg = reporting_cfg.get("reporters", [])
        output_base = self.context.task_dir / "reports" / run_id
        for reporter_cfg in reporters_cfg:
            rtype = reporter_cfg.get("type")
            try:
                reporter = ReporterRegistry.get_reporter(rtype, reporter_cfg, self.context)
                self.logger.info(f"  生成报告: {rtype}")
                # 将分析结果合并后传入
                combined = {"analyses": analysis_results, "raw_metrics": raw_metrics}
                reporter.report(combined, output_base, "clean")
            except ValueError as e:
                self.logger.warning(f"报告器 {rtype} 未注册: {e}")
            except Exception as e:
                self.logger.exception(f"报告器 {rtype} 执行失败: {e}")