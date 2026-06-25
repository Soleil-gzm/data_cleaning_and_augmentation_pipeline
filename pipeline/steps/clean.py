"""
03_clean 步骤：并行清洗（文件级），支持进度条连续更新，自动清理临时配置
"""

import json
import os
import shutil
import re
import subprocess
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

from ..core.step import PipelineStep
from ..analyzers.registry import AnalyzerRegistry
from ..reporters.registry import ReporterRegistry


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
        tag = cfg.get("tag", "default")

        # 生成 run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{timestamp}_clean_{tag}"

        # 并行配置
        global_workers = self.context.config.get("executor", {}).get("max_workers", 1)
        max_workers = cfg.get("max_workers", global_workers)
        if max_workers <= 1:
            self.logger.info("清洗使用串行模式（max_workers=1）")
        else:
            self.logger.info(f"清洗使用并行模式，进程数: {max_workers}")

        # 检查分桶目录
        bucketed_root_path = Path(bucketed_root)
        if not bucketed_root_path.exists():
            self.logger.error(f"分桶目录不存在: {bucketed_root_path}")
            return False

        # 收集所有任务
        all_tasks = []
        bucket_dirs = [d for d in bucketed_root_path.iterdir() if d.is_dir()]
        if not bucket_dirs:
            self.logger.error("分桶目录为空")
            return False

        # 创建带 run_id 的输出目录
        cleaned_base = Path(cleaned_root) / run_id
        trace_base = Path(trace_root) / run_id
        cleaned_base.mkdir(parents=True, exist_ok=True)
        trace_base.mkdir(parents=True, exist_ok=True)

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
                self.logger.warning(f"配置文件 {config_file} 不存在，跳过")
                continue

            output_dir = cleaned_base / bucket_name
            trace_dir = trace_base / bucket_name
            output_dir.mkdir(parents=True, exist_ok=True)
            trace_dir.mkdir(parents=True, exist_ok=True)

            for input_file in bucket_dir.glob("*.jsonl"):
                output_file = output_dir / input_file.name
                file_trace_dir = trace_dir / input_file.stem
                all_tasks.append(
                    {
                        "bucket_name": bucket_name,
                        "input_file": input_file,
                        "output_file": output_file,
                        "trace_dir": file_trace_dir,
                        "config_file": config_file,
                    }
                )

        if not all_tasks:
            self.logger.error("没有找到任何可清洗的文件")
            return False

        self.logger.info(f"共发现 {len(all_tasks)} 个文件待清洗")

        # 执行清洗（串行或并行）
        raw_metrics, success_count, total_tasks = self._run_tasks(
            all_tasks, max_workers
        )

        if success_count == 0:
            self.logger.error("所有文件清洗均失败")
            return False
        self.logger.info(f"清洗完成: {success_count}/{total_tasks} 个文件成功")

        # 转换 defaultdict
        for bucket, stats in raw_metrics["buckets"].items():
            stats["input_turn_dist"] = dict(stats["input_turn_dist"])
            stats["output_turn_dist"] = dict(stats["output_turn_dist"])
        raw_metrics["input_turn_dist"] = dict(raw_metrics["input_turn_dist"])
        raw_metrics["output_turn_dist"] = dict(raw_metrics["output_turn_dist"])
        raw_metrics["buckets"] = dict(raw_metrics["buckets"])

        # 保存原始指标
        report_base = self.context.task_dir / "reports" / run_id
        report_base.mkdir(parents=True, exist_ok=True)
        metrics_path = report_base / "raw_clean_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(raw_metrics, f, indent=2)

        # 保存元数据
        metadata = {
            "run_id": run_id,
            "step": "clean",
            "timestamp": timestamp,
            "tag": tag,
            "input_bucketed_root": str(bucketed_root_path),
            "output_cleaned_root": str(cleaned_base),
            "output_trace_root": str(trace_base),
            "max_workers": max_workers,
            "statistics": {
                "total_input_files": total_tasks,
                "success_files": success_count,
                "failed_files": total_tasks - success_count,
                "total_input_samples": raw_metrics["total_input"],
                "total_output_samples": raw_metrics["total_output"],
                "retention_rate": (
                    raw_metrics["total_output"] / raw_metrics["total_input"]
                    if raw_metrics["total_input"] > 0
                    else 0
                ),
            },
        }
        with open(report_base / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(
            f"清洗 run_id: {run_id} 完成，元数据保存至 {report_base}/run_metadata.json"
        )

        # 触发分析器和报告器
        self._run_analyzers_and_reporters(raw_metrics, run_id)

        return True

    # ================== 任务执行（串行/并行统一） ==================
    def _run_tasks(self, tasks, max_workers):
        """执行任务列表，支持串行/并行，返回 (raw_metrics, success_count, total_tasks)"""
        raw_metrics = {
            "buckets": defaultdict(
                lambda: {
                    "input_samples": 0,
                    "output_samples": 0,
                    "input_turn_dist": defaultdict(int),
                    "output_turn_dist": defaultdict(int),
                }
            ),
            "total_input": 0,
            "total_output": 0,
            "input_turn_dist": defaultdict(int),
            "output_turn_dist": defaultdict(int),
        }
        success_count = 0
        total_tasks = len(tasks)

        if max_workers <= 1:
            # 串行执行
            for task in tqdm(tasks, desc="清洗文件", unit="file"):
                result = self._clean_file_worker(task)
                if result is not None:
                    bucket_name, stats = result
                    self._merge_stats(raw_metrics, bucket_name, stats)
                    success_count += 1
        else:
            # 并行执行
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {
                    executor.submit(self._clean_file_worker, task): task
                    for task in tasks
                }
                with tqdm(total=total_tasks, desc="清洗文件", unit="file") as pbar:
                    for future in as_completed(future_to_task):
                        task = future_to_task[future]
                        try:
                            result = future.result(timeout=600)  # 10分钟超时
                            if result is not None:
                                bucket_name, stats = result
                                self._merge_stats(raw_metrics, bucket_name, stats)
                                success_count += 1
                                pbar.set_postfix(
                                    {
                                        "成功": success_count,
                                        "失败": total_tasks - success_count,
                                    }
                                )
                        except Exception as e:
                            self.logger.error(
                                f"文件 {task['input_file'].name} 清洗异常: {e}"
                            )
                        pbar.update(1)

        return raw_metrics, success_count, total_tasks

    # ================== 合并统计 ==================
    def _merge_stats(self, raw_metrics, bucket_name, stats):
        b_stats = raw_metrics["buckets"][bucket_name]
        b_stats["input_samples"] += stats["input_samples"]
        b_stats["output_samples"] += stats["output_samples"]
        for t, c in stats["input_turn_dist"].items():
            b_stats["input_turn_dist"][t] += c
            raw_metrics["input_turn_dist"][t] += c
        for t, c in stats["output_turn_dist"].items():
            b_stats["output_turn_dist"][t] += c
            raw_metrics["output_turn_dist"][t] += c
        raw_metrics["total_input"] += stats["input_samples"]
        raw_metrics["total_output"] += stats["output_samples"]

    # ================== Worker 函数（静态方法） ==================
    @staticmethod
    def _clean_file_worker(task):
        """
        处理单个文件的清洗任务（在子进程中运行）
        返回: (bucket_name, stats_dict) 或 None（失败）
        """
        bucket_name = task["bucket_name"]
        input_file = task["input_file"]
        output_file = task["output_file"]
        trace_dir = task["trace_dir"]
        config_file = task["config_file"]

        # 确保 trace_dir 存在
        trace_dir.mkdir(parents=True, exist_ok=True)

        # 统计输入文件信息
        input_cnt = CleanStep._count_lines(input_file)
        input_dist = CleanStep._collect_turn_dist(input_file)

        # 生成临时配置文件
        with open(config_file, "r", encoding="utf-8") as f:
            config_content = f.read()
        config_content = config_content.replace(
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
                    new_lines.append(f"work_dir: {trace_dir}")
                else:
                    new_lines.append(line)
            config_content = "\n".join(new_lines)
        else:
            config_content += f"\nwork_dir: {trace_dir}\n"

        # 写入临时配置文件（使用进程ID避免冲突）
        temp_config = Path(f"temp_{input_file.stem}_{os.getpid()}.yaml")
        try:
            with open(temp_config, "w", encoding="utf-8") as f:
                f.write(config_content)

            # 构建命令
            dj_process = shutil.which("dj-process")
            if dj_process is None:
                dj_process = [
                    shutil.which("python"),
                    "-m",
                    "data_juicer.core.process_data",
                ]
            else:
                dj_process = [dj_process]
            cmd = dj_process + ["--config", str(temp_config)]

            # 静默执行
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 单个文件10分钟超时
            )

            if result.returncode != 0 or not output_file.exists():
                return None

            # 统计输出
            output_cnt = CleanStep._count_lines(output_file)
            output_dist = CleanStep._collect_turn_dist(output_file)

            stats = {
                "input_samples": input_cnt,
                "output_samples": output_cnt,
                "input_turn_dist": input_dist,
                "output_turn_dist": output_dist,
            }
            return (bucket_name, stats)

        except subprocess.TimeoutExpired:
            # 超时，直接返回失败
            return None
        except Exception:
            return None
        finally:
            # ===== 关键：确保删除临时文件 =====
            if temp_config.exists():
                try:
                    temp_config.unlink()
                except Exception:
                    pass

    # ================== 辅助静态方法 ==================
    @staticmethod
    def _count_lines(file_path):
        if not file_path.exists():
            return 0
        with open(file_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    @staticmethod
    def _collect_turn_dist(file_path):
        from collections import defaultdict

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

    # ================== 桶配置匹配 ==================
    def _get_config_for_bucket(self, bucket_name, bucket_config_map):
        for entry in bucket_config_map:
            pattern = entry.get("pattern")
            if pattern and re.match(pattern, bucket_name):
                return entry.get("config")
        return None

    # ================== 分析器和报告器触发 ==================
    def _run_analyzers_and_reporters(self, raw_metrics, run_id):
        reporting_cfg = self.context.config.get("reporting", {})
        if not reporting_cfg.get("enabled", True):
            return

        step_cfg = self.context.get_step_config("03_clean")
        analyzer_names = step_cfg.get(
            "attach_analyzers", ["RetentionAnalyzer", "TurnDistributionAnalyzer"]
        )

        analysis_results = {}
        for name in analyzer_names:
            try:
                analyzer = AnalyzerRegistry.get_analyzer(name, self.context)
                self.logger.info(f"  运行分析器: {name}")
                analysis_results[name] = analyzer.analyze(raw_metrics)
            except Exception as e:
                self.logger.exception(f"分析器 {name} 执行失败: {e}")

        if not analysis_results:
            return

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
            except Exception as e:
                self.logger.exception(f"报告器 {rtype} 执行失败: {e}")
