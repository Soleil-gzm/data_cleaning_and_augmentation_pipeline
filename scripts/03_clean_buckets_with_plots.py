#!/usr/bin/env python3
"""
分桶清洗脚本（带统计和可视化）
支持 --tag 参数，输出目录格式：cleaned_jsonl/{timestamp}_clean_{tag}/
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import logging
import re

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# 尝试导入绘图库
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib 未安装，将无法生成图表。")

# ========== 配置 ==========
BUCKETED_ROOT = "task_20260429/task_92049/output_cleaning/bucketed"
CLEANED_ROOT = "task_20260429/task_92049/output_cleaning/cleaned_jsonl"
TRACE_ROOT = "task_20260429/task_92049/output_cleaning/trace_output"
CONFIGS_DIR = "configs/configs_qa"
REPORT_DIR = "task_20260429/task_92049/output_cleaning/cleaning_reports"

PLOT_TURNS = None

# BUCKET_CONFIG_MAP = {
#     "bucket_0": "overal_config.yaml",
#     "bucket_1": "config_bucket_1.yaml",
#     "bucket_2": "overal_config.yaml",
#     "bucket_3": "overal_config.yaml",
#     "bucket_4": "overal_config.yaml",
#     "bucket_5": "overal_config.yaml",
#     "bucket_6": "overal_config.yaml",
#     "bucket_7": "overal_config.yaml",
#     "bucket_8": "overal_config.yaml",
#     "bucket_9": "overal_config.yaml",
#     "bucket_10plus": "config_bucket_10plus.yaml",
# }

# task_20260429/task_92049/
BUCKET_CONFIG_MAP = {
    "bucket_0": "overal_config.yaml",
    "bucket_1": "overal_config.yaml",
    "bucket_2": "overal_config.yaml",
    "bucket_3": "overal_config.yaml",
    "bucket_4": "overal_config.yaml",
    "bucket_5": "overal_config.yaml",
    "bucket_6": "overal_config.yaml",
    "bucket_7": "overal_config.yaml",
    "bucket_8": "overal_config.yaml",
    "bucket_9": "overal_config.yaml",
    "bucket_10": "overal_config.yaml",
    "bucket_11": "overal_config.yaml",
    "bucket_12": "overal_config.yaml",
    "bucket_13_22": "overal_config.yaml",
    "bucket_23plus": "config_bucket_10plus.yaml",
}

# ========== 辅助函数 ==========
def setup_logger(task_dir):
    log_dir = task_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "clean_buckets.log"
    logger = logging.getLogger("CleanBuckets")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def get_config_for_bucket(bucket_name, bucket_config_map):
    """
    根据正则表达式匹配桶的配置文件
    bucket_config_map: list of {"pattern": "regex", "config": "filename.yaml"}
    """
    for entry in bucket_config_map:
        pattern = entry.get('pattern')
        if pattern is None:
            continue
        if re.match(pattern, bucket_name):
            return entry['config']
    return None

def count_samples_in_jsonl(file_path):
    if not file_path.exists():
        return 0
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def collect_turn_distribution(file_path):
    dist = defaultdict(int)
    if not file_path.exists():
        return dist
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                turn = data.get('turn')
                if turn is not None:
                    dist[turn] += 1
            except:
                pass
    return dist

def plot_turn_distribution(bucket_name, input_dist, output_dist, output_dir, selected_turns=None):
    if not HAS_MATPLOTLIB:
        return
    
    # 强制使用文泉驿正黑字体，避免中文警告
    import matplotlib.font_manager as fm
    font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'WenQuanYi Zen Hei'
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

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
    plt.bar(x, input_counts, width, label='清洗前', color='steelblue')
    plt.bar([i + width for i in x], output_counts, width, label='清洗后', color='salmon')
    plt.xlabel('轮次 (turn)')
    plt.ylabel('样本数量')
    plt.title(f'{bucket_name} 清洗前后轮次分布对比')
    plt.xticks([i + width/2 for i in x], all_turns, rotation=45)
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / f'{bucket_name}_turn_distribution.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"    图表已保存: {plot_path}")

def clean_bucket(bucket_dir, config_file, output_dir, trace_dir, stats):
    if not bucket_dir.exists():
        logger.info(f"  目录不存在: {bucket_dir}")
        return 0
    input_files = list(bucket_dir.glob("*.jsonl"))
    if not input_files:
        logger.info(f"  没有找到 JSONL 文件")
        return 0
    print(f"\n  发现 {len(input_files)} 个文件")
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    bucket_stats = {
        "input_samples": 0,
        "output_samples": 0,
        "input_turn_dist": defaultdict(int),
        "output_turn_dist": defaultdict(int),
    }
    success_count = 0

    # 获取 dj-process 可执行文件路径
    dj_process = shutil.which('dj-process')
    if dj_process is None:
        # 尝试使用 python -m 方式
        dj_process = [sys.executable, '-m', 'data_juicer.core.process_data']
    else:
        dj_process = [dj_process]

    for input_file in input_files:
        output_file = output_dir / input_file.name
        trace_subdir = trace_dir / input_file.stem
        input_cnt = count_samples_in_jsonl(input_file)
        input_turn_dist = collect_turn_distribution(input_file)

        # 读取配置模板
        with open(config_file, 'r', encoding='utf-8') as f:
            config_content = f.read()
        config_content = config_content.replace('__INPUT_FILE__', str(input_file.absolute()))
        config_content = config_content.replace('__OUTPUT_FILE__', str(output_file.absolute()))

        if 'work_dir:' in config_content:
            lines = config_content.splitlines()
            new_lines = []
            for line in lines:
                if line.strip().startswith('work_dir:'):
                    new_lines.append(f"work_dir: {trace_subdir}")
                else:
                    new_lines.append(line)
            config_content = '\n'.join(new_lines)
        else:
            config_content += f"\nwork_dir: {trace_subdir}\n"

        temp_config = Path(f"temp_{input_file.stem}.yaml")
        with open(temp_config, 'w', encoding='utf-8') as f:
            f.write(config_content)

        # 构建命令列表
        cmd = dj_process + ['--config', str(temp_config)]
        # 使用当前环境变量运行子进程
        result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)

        if temp_config.exists():
            temp_config.unlink()

        if result.returncode == 0:
            if output_file.exists():
                output_cnt = count_samples_in_jsonl(output_file)
                output_turn_dist = collect_turn_distribution(output_file)
                print(f"    ✅ {input_file.name}: {input_cnt} → {output_cnt} 条")
            else:
                output_cnt = 0
                output_turn_dist = defaultdict(int)
                print(f"    ⚠️ {input_file.name}: 清洗后无输出（所有样本可能被过滤）")
            bucket_stats["input_samples"] += input_cnt
            bucket_stats["output_samples"] += output_cnt
            for turn, cnt in input_turn_dist.items():
                bucket_stats["input_turn_dist"][turn] += cnt
            for turn, cnt in output_turn_dist.items():
                bucket_stats["output_turn_dist"][turn] += cnt
            success_count += 1
        else:
            print(f"    ❌ {input_file.name} 清洗失败，退出码 {result.returncode}")
            print(f"        stdout: {result.stdout[:500]}")
            print(f"        stderr: {result.stderr[:500]}")
            error_log = trace_subdir / "error.log"
            error_log.parent.mkdir(parents=True, exist_ok=True)
            with open(error_log, 'w') as f:
                f.write(f"Return code: {result.returncode}\n")
                f.write("STDOUT:\n" + result.stdout)
                f.write("\nSTDERR:\n" + result.stderr)
            print(f"        完整错误已保存到 {error_log}")

    retention_rate = 0.0
    if bucket_stats["input_samples"] > 0:
        retention_rate = bucket_stats["output_samples"] / bucket_stats["input_samples"]
    stats["buckets"][bucket_dir.name] = {
        "input_samples": bucket_stats["input_samples"],
        "output_samples": bucket_stats["output_samples"],
        "retention_rate": retention_rate,
        "input_turn_dist": dict(bucket_stats["input_turn_dist"]),
        "output_turn_dist": dict(bucket_stats["output_turn_dist"]),
    }
    return success_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, help="全局配置JSON")
    parser.add_argument("--bucketed_root", type=str, help="分桶后的根目录")
    parser.add_argument("--cleaned_root", type=str, help="清洗输出根目录")
    parser.add_argument("--trace_root", type=str, help="trace输出根目录")
    parser.add_argument("--configs_dir", type=str, default="configs/configs_qa")
    parser.add_argument("--bucket_config_map", type=str, help="JSON格式的正则映射列表")
    args = parser.parse_args()
    
    if args.config_json:
        config = json.loads(args.config_json)
        task_name = config['task_name']
        base_dir = Path(config['paths']['output']['base_dir'])
        bucketed_root = base_dir / task_name / "bucketed"
        cleaned_root = base_dir / task_name / "cleaned_jsonl"
        trace_root = base_dir / task_name / "trace_output"
        configs_dir = args.configs_dir or config.get('steps', {}).get('03_clean', {}).get('configs_dir', 'configs/configs_qa')
        bucket_config_map = json.loads(args.bucket_config_map) if args.bucket_config_map else config.get('steps', {}).get('03_clean', {}).get('bucket_config_map', [])
    else:
        # 兼容直接运行
        bucketed_root = Path(args.bucketed_root)
        cleaned_root = Path(args.cleaned_root)
        trace_root = Path(args.trace_root)
        configs_dir = Path(args.configs_dir)
        bucket_config_map = json.loads(args.bucket_config_map) if args.bucket_config_map else []
    
    # 设置日志
    task_dir = bucketed_root.parent
    logger = setup_logger(task_dir)
    logger.info("开始清洗流程")
    logger.info(f"分桶根目录: {bucketed_root}")
    logger.info(f"清洗输出根目录: {cleaned_root}")
    
    # 创建时间戳 run_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_clean_{config.get('task_name', 'task')}"
    cleaned_base = cleaned_root / run_id
    trace_base = trace_root / run_id
    report_base = task_dir / "reports" / run_id
    cleaned_base.mkdir(parents=True, exist_ok=True)
    trace_base.mkdir(parents=True, exist_ok=True)
    report_base.mkdir(parents=True, exist_ok=True)
    
    overall_stats = {"buckets": {}, "total_input": 0, "total_output": 0}
    
    # 遍历所有桶目录
    for bucket_dir in bucketed_root.iterdir():
        if not bucket_dir.is_dir():
            continue
        bucket_name = bucket_dir.name
        # 获取该桶对应的配置文件
        config_filename = get_config_for_bucket(bucket_name, bucket_config_map)
        if not config_filename:
            logger.warning(f"桶 {bucket_name} 未匹配到配置文件，跳过")
            continue
        config_file = Path(configs_dir) / config_filename
        if not config_file.exists():
            logger.warning(f"配置文件 {config_file} 不存在，跳过桶 {bucket_name}")
            continue
        
        logger.info(f"处理桶: {bucket_name}，使用配置 {config_filename}")
        output_dir = cleaned_base / bucket_name
        trace_dir = trace_base / bucket_name



    # parser = argparse.ArgumentParser()
    # parser.add_argument("--tag", type=str, default="default", help="清洗任务标签")
    # args = parser.parse_args()
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # run_id = f"{timestamp}_clean_{args.tag}"
    # cleaned_base = Path(CLEANED_ROOT) / run_id
    # trace_base = Path(TRACE_ROOT) / run_id
    # report_base = Path(REPORT_DIR) / run_id
    # report_base.mkdir(parents=True, exist_ok=True)
    # print(f"Run ID: {run_id}")
    # print(f"清洗结果目录: {cleaned_base}")
    # print(f"Trace 目录: {trace_base}")
    # print(f"报告目录: {report_base}")
    # overall_stats = {
    #     "run_id": run_id,
    #     "buckets": {},
    #     "total_input": 0,
    #     "total_output": 0,
    #     "overall_input_turn_dist": defaultdict(int),
    #     "overall_output_turn_dist": defaultdict(int),
    # }
    # total_success = 0
    # for bucket_name, config_filename in BUCKET_CONFIG_MAP.items():
    #     bucket_dir = Path(BUCKETED_ROOT) / bucket_name
    #     if not bucket_dir.exists():
    #         print(f"跳过不存在的桶目录: {bucket_dir}")
    #         continue
    #     config_file = Path(CONFIGS_DIR) / config_filename
    #     if not config_file.exists():
    #         print(f"⚠️ 配置文件 {config_file} 不存在，跳过桶 {bucket_name}")
    #         continue
    #     print(f"\n处理桶: {bucket_name}")
    #     output_dir = cleaned_base / bucket_name
    #     trace_dir = trace_base / bucket_name
    #     success = clean_bucket(bucket_dir, config_file, output_dir, trace_dir, overall_stats)
    #     total_success += success
    #     if bucket_name in overall_stats["buckets"]:
    #         bucket_stats = overall_stats["buckets"][bucket_name]
    #         bucket_report_file = report_base / f"{bucket_name}_report.json"
    #         with open(bucket_report_file, 'w') as f:
    #             json.dump(bucket_stats, f, indent=2)
    #         plot_turn_distribution(
    #             bucket_name,
    #             bucket_stats["input_turn_dist"],
    #             bucket_stats["output_turn_dist"],
    #             report_base,
    #             selected_turns=PLOT_TURNS
    #         )
    # for bucket, stats in overall_stats["buckets"].items():
    #     overall_stats["total_input"] += stats["input_samples"]
    #     overall_stats["total_output"] += stats["output_samples"]
    #     for turn, cnt in stats["input_turn_dist"].items():
    #         overall_stats["overall_input_turn_dist"][turn] += cnt
    #     for turn, cnt in stats["output_turn_dist"].items():
    #         overall_stats["overall_output_turn_dist"][turn] += cnt
    
    # 保存全局报告
    overall_report_file = report_base / "overall_report.json"
    report_data = {
        "run_id": overall_stats["run_id"],
        "buckets": overall_stats["buckets"],
        "total_input": overall_stats["total_input"],
        "total_output": overall_stats["total_output"],
        "overall_input_turn_dist": dict(overall_stats["overall_input_turn_dist"]),
        "overall_output_turn_dist": dict(overall_stats["overall_output_turn_dist"]),
    }
    with open(overall_report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    plot_turn_distribution(
        "overall",
        overall_stats["overall_input_turn_dist"],
        overall_stats["overall_output_turn_dist"],
        report_base,
        selected_turns=PLOT_TURNS
    )
    # 保存 CSV
    summary_csv = report_base / "bucket_summary.csv"
    with open(summary_csv, 'w', encoding='utf-8') as f:
        f.write("bucket_name,input_samples,output_samples,retention_rate\n")
        for bucket, stats in overall_stats["buckets"].items():
            inp = stats["input_samples"]
            out = stats["output_samples"]
            rate = stats.get("retention_rate", out/inp if inp>0 else 0)
            f.write(f"{bucket},{inp},{out},{rate:.6f}\n")
    csv_file = report_base / "turn_distribution_comparison.csv"
    with open(csv_file, 'w') as f:
        f.write("bucket,turn,input_count,output_count\n")
        for bucket, stats in overall_stats["buckets"].items():
            input_dist = stats["input_turn_dist"]
            output_dist = stats["output_turn_dist"]
            all_turns = set(input_dist.keys()) | set(output_dist.keys())
            for turn in sorted(all_turns):
                in_cnt = input_dist.get(turn, 0)
                out_cnt = output_dist.get(turn, 0)
                f.write(f"{bucket},{turn},{in_cnt},{out_cnt}\n")
    # 写入元数据
    metadata = {
        "run_id": run_id,
        "task": "clean",
        "start_time": datetime.now().isoformat(),
        "command_line": " ".join(sys.argv),
        "config": {
            "buckets_config_map": BUCKET_CONFIG_MAP,
            "configs_dir": CONFIGS_DIR,
            "plot_turns": PLOT_TURNS,
        },
        "statistics": {
            "total_input_samples": overall_stats["total_input"],
            "total_output_samples": overall_stats["total_output"],
            "retention_rate": overall_stats["total_output"] / overall_stats["total_input"] if overall_stats["total_input"] > 0 else 0,
            "buckets": overall_stats["buckets"],
        },
        "output_dirs": {
            "cleaned_jsonl": str(cleaned_base),
            "trace": str(trace_base),
            "reports": str(report_base),
        }
    }
    metadata_path = cleaned_base / "run_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    # 打印汇总
    print("\n" + "="*60)
    print("清洗统计汇总")
    print("="*60)
    print(f"Run ID: {run_id}")
    print(f"总体输入样本数: {overall_stats['total_input']}")
    print(f"总体输出样本数: {overall_stats['total_output']}")
    if overall_stats['total_input'] > 0:
        print(f"总体保留率: {overall_stats['total_output']/overall_stats['total_input']*100:.2f}%")
    else:
        print("总体保留率: N/A (无输入样本)")
    print("\n各桶统计:")
    for bucket, stats in overall_stats["buckets"].items():
        inp = stats["input_samples"]
        out = stats["output_samples"]
        rate = out/inp*100 if inp>0 else 0
        print(f"  {bucket}: {inp} → {out} ({rate:.2f}%)")
    print(f"\n报告已保存到: {report_base}")
    print(f"清洗完成！")

if __name__ == "__main__":
    main()