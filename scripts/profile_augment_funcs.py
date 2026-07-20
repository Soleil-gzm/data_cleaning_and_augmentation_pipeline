#!/usr/bin/env python3
"""
独立的增强函数性能统计脚本
使用全局累积性能监控方案，对各增强函数进行单独 benchmark
无需运行完整流水线即可快速定位性能瓶颈
"""

import sys
import random
from pathlib import Path

scripts_dir = str(Path(__file__).parent)
sys.path.append(scripts_dir)

from common import augment_utils_add as aug_utils

TEST_SENTENCES = [
    "我现在工资，每个月倒是可以给您发，但是每",
    "打通不的话就挂了办",
    "华夏银行信用卡逾期了怎么办",
    "最低还款额是多少能不能减免",
    "我想协商一下还款计划利息太高了",
    "你好我想咨询一下贷款的事情",
    "这个月的账单还没有收到",
    "能不能帮我查一下账户余额",
    "密码忘记了怎么找回",
    "申请信用卡需要什么材料",
]

def run_single_func_profiling(func_name, sentences, num_runs=100):
    """对单个增强函数进行多次调用，收集性能统计"""
    aug_utils.reset_augment_perf_stats()
    
    func = aug_utils.AUGMENT_FUNC_MAP.get(func_name)
    if func is None:
        print(f"错误：未找到函数 {func_name}")
        return None
    
    for i in range(num_runs):
        sent = sentences[i % len(sentences)]
        func(sent)
    
    stats = aug_utils.get_augment_perf_stats()
    return stats

def run_multi_step_profiling(sentences, num_runs=100, min_steps=1, max_steps=3, weights=None):
    """对 multi_step_augment 进行多次调用，收集各函数的累积统计"""
    aug_utils.reset_augment_perf_stats()
    
    for i in range(num_runs):
        sent = sentences[i % len(sentences)]
        aug_utils.multi_step_augment(sent, min_steps=min_steps, max_steps=max_steps, weights=weights)
    
    stats = aug_utils.get_augment_perf_stats()
    return stats

def main():
    print("=" * 70)
    print("增强函数独立性能统计")
    print("=" * 70)
    
    print("\n[1] 单个函数性能测试")
    print("-" * 70)
    print(f"测试句子数: {len(TEST_SENTENCES)}")
    print(f"每个函数测试次数: 100")
    print()
    
    results = {}
    for func_name in aug_utils.AUGMENT_FUNC_MAP.keys():
        stats = run_single_func_profiling(func_name, TEST_SENTENCES, num_runs=100)
        if stats:
            results[func_name] = {
                "total_time": stats["elapsed"].get(func_name, 0),
                "calls": stats["calls"].get(func_name, 0),
                "avg_time": stats["elapsed"].get(func_name, 0) / stats["calls"].get(func_name, 1)
            }
    
    print(f"{'函数名':<25} {'调用次数':>10} {'总耗时(s)':>12} {'平均耗时(ms)':>14}")
    print("-" * 70)
    for name, r in sorted(results.items(), key=lambda x: x[1]["avg_time"], reverse=True):
        print(f"{name:<25} {r['calls']:>10} {r['total_time']:>12.4f} {r['avg_time']*1000:>14.2f}")
    
    print("\n" + "=" * 70)
    print("[2] 多步叠加增强性能测试（模拟真实流水线）")
    print("=" * 70)
    print(f"测试句子数: {len(TEST_SENTENCES)}")
    print(f"测试次数: 500")
    print(f"增强步数: 1-3 随机")
    print()
    
    multi_stats = run_multi_step_profiling(TEST_SENTENCES, num_runs=500, min_steps=1, max_steps=3)
    aug_utils.print_augment_perf_stats()
    
    print("\n" + "=" * 70)
    print("[3] 带权重配置的多步增强性能测试")
    print("=" * 70)
    
    test_weights_list = [
        {"asr_noise": 5.0, "homophone": 2.0, "insert_filler": 1.0},
        {"similarword": 3.0, "random_entity_replace": 2.0},
        {"stutter": 1.0, "word_repetition": 1.0, "reorder": 1.0},
    ]
    
    for idx, weights in enumerate(test_weights_list, 1):
        print(f"\n配置 {idx}: {weights}")
        aug_utils.reset_augment_perf_stats()
        
        for i in range(200):
            sent = TEST_SENTENCES[i % len(TEST_SENTENCES)]
            aug_utils.multi_step_augment(sent, min_steps=2, max_steps=3, weights=weights)
        
        stats = aug_utils.get_augment_perf_stats()
        total = stats["total_time"]
        
        print(f"{'函数名':<25} {'调用次数':>10} {'总耗时(s)':>12} {'占比':>8}")
        print("-" * 60)
        for name, calls in sorted(stats["calls"].items(), key=lambda x: stats["elapsed"].get(x[0], 0), reverse=True):
            elapsed = stats["elapsed"].get(name, 0)
            pct = (elapsed / total) * 100 if total > 0 else 0
            print(f"{name:<25} {calls:>10} {elapsed:>12.4f} {pct:>8.1f}%")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()