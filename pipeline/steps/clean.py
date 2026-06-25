"""
03_clean 步骤：调用 Data-Juicer 清洗每个桶
支持多进程并行（可配置），进度条统一由主进程控制
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

from tqdm import tqdm

from ..core.step import PipelineStep
from ..analyzers.registry import AnalyzerRegistry
from ..reporters.registry import ReporterRegistry


# ========== Worker 函数（在子进程中运行） ==========
def clean_bucket_worker(
    bucket_dir: Path,
    config_file: Path,
    cleaned_base: Path,
    trace_base: Path,
    dj_process: List[str],
) -> Tuple[str, Dict[str, Any]]:
    """
    子进程函数：清洗单个桶的所有 JSONL 文件。
    返回 (bucket_name, stats_dict)
    """
    bucket_name = bucket_dir.name
    output_dir = cleaned_base / bucket_name
    trace_dir = trace_base / bucket_name
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    input_files = list(bucket_dir.glob("*.jsonl"))
    bucket_stats = {
        "input_samples": 0,
        "output_samples": 0,
        "input_turn_dist": defaultdict(int),
        "output_turn_dist": defaultdict(int),
    }

    # 读取配置文件内容（每个桶可能不同，但此处传入的是单一配置）
    with open(config_file, "r", encoding="utf-8") as f:
        config_template = f.read()

    for input_file in input_files:
        output_file = output_dir / input_file.name
        trace_subdir = trace_dir / input_file.stem

        # 统计输入
        input_cnt = _count_lines(input_file)
        input_dist = _collect_turn_dist(input_file)

        # 生成临时配置
        config_content = config_template.replace(
            "__INPUT_FILE__", str(input_file.absolute())
        )
        config_content = config_content.replace(
            "__OUTPUT_FILE__", str(output_file.absolute())
        )
        # 替换 work_dir
        if "work_dir:" in config_content:
            lines = config_content.splitlines()
            new_lines = []
            for line in lines:
                if line.strip().startswith("work_dir:"):
                    new_lines.append(f"work_dir: {trace_subdir}")
                else:
                    new_lines.append(line)
            config_content = "\n".join(new_lines)
        else:
            config_content += f"\nwork_dir: {trace_subdir}\n"

        temp_config = Path(f"temp_{input_file.stem}_{os.getpid()}.yaml")
        with open(temp_config, "w", encoding="utf-8") as f:
            f.write(config_content)

        # 执行 Data-Juicer
        cmd = dj_process + ["--config", str(temp_config)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # 清理临时文件
        try:
            temp_config.unlink()
        except:
            pass

        if result.returncode == 0 and output_file.exists():
            output_cnt = _count_lines(output_file)
            output_dist = _collect_turn_dist(output_file)
        else:
            output_cnt = 0
            output_dist = defaultdict(int)

        # 累加统计
        bucket_stats["input_samples"] += input_cnt
        bucket_stats["output_samples"] += output_cnt
        for t, c in input_dist.items():
            bucket_stats["input_turn_dist"][t] += c
        for t, c in output_dist.items():
            bucket_stats["output_turn_dist"][t] += c

    # 将 defaultdict 转为普通 dict
    bucket_stats["input_turn_dist"] = dict(bucket_stats["input_turn_dist"])
    bucket_stats["output_turn_dist"] = dict(bucket_stats["output_turn_dist"])
    return bucket_name, bucket_stats


def _count_lines(file_path: Path) -> int:
    if not file_path.exists():
        return 0
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _collect_turn_dist(file_path: Path) -> Dict[int, int]:
    dist = defaultdict(int)
    if not file_path.exists():
        return dist
    with open(file_path, "r", encoding="utf-8") as f:
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


# ========== 步骤类 ==========
class CleanStep(PipelineStep):
    def run(self) -> bool:
        cfg = self.context.get_step_config("03_clean")
        bucketed_root = cfg.get("bucketed_root") or (self.context.task_dir / "bucketed")
        cleaned_root = cfg.get("cleaned_root") or (
            self.context.task_dir / "cleaned_jsonl"
        )
        trace_root = cfg.get("trace_root") or (self.context.task_dir / "trace_output")
        configs_dir = Path(cfg.get("configs_dir", "configs/configs_qa"))
        bucket_config_map = cfg.get("bucket_config_map", [])

        # 确定并行度
        global_workers = self.context.config.get("executor", {}).get("max_workers", 1)
        executor_type = self.context.config.get("executor", {}).get(
            "type", "sequential"
        )
        max_workers = cfg.get("max_workers", global_workers)
        # 如果全局类型为 sequential，则强制串行
        if executor_type == "sequential":
            max_workers = 1

        self.logger.info(f"清洗步骤并行度: {max_workers}")

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
        if not bucket_dirs:
            self.logger.error(f"分桶目录为空: {bucketed_root_path}")
            return False

        # 确定 Data-Juicer 命令
        dj_process = shutil.which("dj-process")
        if dj_process is None:
            dj_process = [shutil.which("python"), "-m", "data_juicer.core.process_data"]
        else:
            dj_process = [dj_process]

        # 验证 Data-Juicer 可用性（仅第一次）
        try:
            test_cmd = dj_process + ["--help"]
            subprocess.run(test_cmd, capture_output=True, timeout=10, check=True)
        except Exception as e:
            self.logger.error(f"Data-Juicer 不可用: {e}")
            return False

        # 为每个桶匹配配置文件
        tasks = []
        for bucket_dir in bucket_dirs:
            bucket_name = bucket_dir.name
            config_filename = self._get_config_for_bucket(
                bucket_name, bucket_config_map
            )
            if not config_filename:
                self.logger.warning(f"桶 {bucket_name} 未匹配配置文件，跳过")
                continue
            config_file = configs_dir / config_filename
            if not config_file.exists():
                self.logger.warning(
                    f"配置文件 {config_file} 不存在，跳过桶 {bucket_name}"
                )
                continue
            tasks.append((bucket_dir, config_file))

        if not tasks:
            self.logger.error("没有可清洗的桶（匹配不到配置文件）")
            return False

        self.logger.info(f"开始清洗 {len(tasks)} 个桶，使用 {max_workers} 个进程")

        # ===== 并行或串行执行 =====
        raw_metrics = {
            "buckets": {},
            "total_input": 0,
            "total_output": 0,
            "input_turn_dist": defaultdict(int),
            "output_turn_dist": defaultdict(int),
        }

        if max_workers <= 1:
            # 串行执行（保留原有逻辑，便于调试）
            for bucket_dir, config_file in tasks:
                bucket_name, stats = clean_bucket_worker(
                    bucket_dir, config_file, cleaned_base, trace_base, dj_process
                )
                self._aggregate_stats(raw_metrics, bucket_name, stats)
                self.logger.info(
                    f"  桶 {bucket_name} 完成: 输入 {stats['input_samples']}，输出 {stats['output_samples']}"
                )
        else:
            # 并行执行：使用进程池
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务，获取 future 字典
                future_to_bucket = {
                    executor.submit(
                        clean_bucket_worker,
                        bucket_dir,
                        config_file,
                        cleaned_base,
                        trace_base,
                        dj_process,
                    ): bucket_dir.name
                    for bucket_dir, config_file in tasks
                }

                # 进度条（主进程控制）
                with tqdm(total=len(tasks), desc="清洗桶", unit="bucket") as pbar:
                    for future in as_completed(future_to_bucket):
                        bucket_name = future_to_bucket[future]
                        try:
                            _, stats = future.result()
                            self._aggregate_stats(raw_metrics, bucket_name, stats)
                            # 更新进度条
                            pbar.update(1)
                            pbar.set_postfix({"当前桶": bucket_name})
                            # 也可输出日志（但避免过于频繁）
                            self.logger.info(
                                f"  桶 {bucket_name} 完成: 输入 {stats['input_samples']}，输出 {stats['output_samples']}"
                            )
                        except Exception as e:
                            self.logger.error(f"桶 {bucket_name} 清洗失败: {e}")
                            # 进度条仍更新，但标记失败（可选）
                            pbar.update(1)
                            pbar.set_postfix({"失败桶": bucket_name})

        # 转换 defaultdict 为 dict
        raw_metrics["input_turn_dist"] = dict(raw_metrics["input_turn_dist"])
        raw_metrics["output_turn_dist"] = dict(raw_metrics["output_turn_dist"])

        # 检查是否有有效输出
        if raw_metrics["total_output"] == 0:
            self.logger.error("所有桶清洗后均无输出，请检查 Data-Juicer 配置")
            return False

        # 保存原始指标
        metrics_path = (
            self.context.task_dir / "reports" / run_id / "raw_clean_metrics.json"
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(raw_metrics, f, indent=2)
        self.logger.info(f"原始清洗指标已保存: {metrics_path}")

        # ===== 触发分析器和报告器 =====
        self._run_analyzers_and_reporters(raw_metrics, run_id)

        self.logger.info(
            f"清洗完成: 总输入 {raw_metrics['total_input']}，总输出 {raw_metrics['total_output']}"
        )
        return True

    def _aggregate_stats(self, raw_metrics: Dict, bucket_name: str, stats: Dict):
        """将单个桶的统计合并到总指标中"""
        raw_metrics["buckets"][bucket_name] = stats
        raw_metrics["total_input"] += stats["input_samples"]
        raw_metrics["total_output"] += stats["output_samples"]
        for t, c in stats["input_turn_dist"].items():
            raw_metrics["input_turn_dist"][t] += c
        for t, c in stats["output_turn_dist"].items():
            raw_metrics["output_turn_dist"][t] += c

    def _get_config_for_bucket(
        self, bucket_name: str, bucket_config_map: List[Dict]
    ) -> str:
        for entry in bucket_config_map:
            pattern = entry.get("pattern")
            if pattern and re.match(pattern, bucket_name):
                return entry.get("config")
        return None

    def _run_analyzers_and_reporters(self, raw_metrics: dict, run_id: str):
        """触发配置中绑定的分析器和报告器"""
        reporting_cfg = self.context.config.get("reporting", {})
        if not reporting_cfg.get("enabled", True):
            self.logger.info("报告模块已禁用")
            return

        # 获取该步骤绑定的分析器列表
        step_cfg = self.context.get_step_config("03_clean")
        analyzer_names = step_cfg.get(
            "attach_analyzers", ["RetentionAnalyzer", "TurnDistributionAnalyzer"]
        )

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
                reporter = ReporterRegistry.get_reporter(
                    rtype, reporter_cfg, self.context
                )
                self.logger.info(f"  生成报告: {rtype}")
                combined = {"analyses": analysis_results, "raw_metrics": raw_metrics}
                reporter.report(combined, output_base, "clean")
            except ValueError as e:
                self.logger.warning(f"报告器 {rtype} 未注册: {e}")
            except Exception as e:
                self.logger.exception(f"报告器 {rtype} 执行失败: {e}")
