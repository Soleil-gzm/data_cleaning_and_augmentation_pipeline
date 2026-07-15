#!/usr/bin/env python3
"""
测量所有增强函数的单次耗时，找出真正的瓶颈
"""

import sys
import time
import random
from pathlib import Path

scripts_dir = str(Path(__file__).parent)
sys.path.append(scripts_dir)

from common import augment_utils_add as aug_utils

# 先设置 ASR augmenter
from common.asr_noise_augmenter import AsrNoiseAugmenter

project_root = Path(__file__).parent.parent
vectors_path = project_root / "resources" / "prev_clean" / "sample_20" / "qwen" / "prev_clean_prev_window_1_no_prob" / "abnormal_vectors.pkl"
pinyin_path = project_root / "resources" / "prev_clean" / "sample_20" / "qwen" / "prev_clean_prev_window_1_no_prob" / "abnormal_pinyin.pkl"
prev_map_path = project_root / "resources" / "prev_clean" / "sample_20" / "qwen" / "prev_clean_prev_window_1_no_prob" / "prev_to_abnormals.pkl"
model_path = project_root / "Models" / "paraphrase-multilingual-MiniLM-L12-v2"

augmenter = AsrNoiseAugmenter(
    vectors_path=str(vectors_path),
    pinyin_path=str(pinyin_path),
    prev_map_path=str(prev_map_path),
    model_path=str(model_path)
)
aug_utils.set_asr_augmenter(augmenter)


# 测试句子
TEST_SENTENCES = [
    "我现在工资，每个月倒是可以给您发，但是每",
    "打通不的话就挂了办",
    "华夏银行信用卡逾期了怎么办",
    "最低还款额是多少能不能减免",
    "我想协商一下还款计划利息太高了",
]

# 所有增强函数
AUG_FUNCS = {
    "insert_filler": aug_utils.apply_insert_filler,
    "stutter": aug_utils.apply_stutter,
    "reorder": aug_utils.apply_reorder,
    "homophone": aug_utils.apply_homophone,
    "random_delete": aug_utils.apply_random_delete,
    "random_entity_replace": aug_utils.apply_random_entity_replace,
    "similarword": aug_utils.apply_similarword,
    "word_repetition": aug_utils.apply_word_repetition,
    "asr_noise": aug_utils.apply_asr_noise,
}


def benchmark_single_function(func_name, func, sentences, num_runs=20):
    """对单个函数进行多次测试"""
    times = []
    for i in range(num_runs):
        sent = sentences[i % len(sentences)]
        start = time.time()
        try:
            result = func(sent)
        except Exception as e:
            result = f"ERROR: {e}"
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)
    return avg, min_t, max_t, times


def main():
    print("=" * 80)
    print("所有增强函数性能基准测试")
    print("=" * 80)
    print(f"测试句子数: {len(TEST_SENTENCES)}")
    print(f"每个函数测试次数: 20")
    print()
    
    # 预热（第一次调用可能有初始化开销）
    print("预热中...")
    for name, func in AUG_FUNCS.items():
        try:
            func(TEST_SENTENCES[0])
        except:
            pass
    print("预热完成\n")
    
    results = {}
    
    print(f"{'函数名':<25} {'平均耗时':>10} {'最小':>10} {'最大':>10} {'示例输出'}")
    print("-" * 80)
    
    for name, func in AUG_FUNCS.items():
        avg, min_t, max_t, times = benchmark_single_function(name, func, TEST_SENTENCES)
        results[name] = {"avg": avg, "min": min_t, "max": max_t}
        
        # 获取一个示例输出
        try:
            example = func(TEST_SENTENCES[0])[:30]
        except:
            example = "ERROR"
        
        print(f"{name:<25} {avg:>9.4f}s {min_t:>9.4f}s {max_t:>9.4f}s  {example}")
    
    print("\n" + "=" * 80)
    print("瓶颈分析")
    print("=" * 80)
    
    # 按耗时排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]["avg"], reverse=True)
    
    print(f"\n{'排名':<5} {'函数名':<25} {'平均耗时':>10} {'占比':>8}")
    print("-" * 50)
    total_avg = sum(r["avg"] for r in results.values())
    for rank, (name, r) in enumerate(sorted_results, 1):
        pct = r["avg"] / total_avg * 100
        bar = "█" * int(pct / 3)
        print(f"{rank:<5} {name:<25} {r['avg']:>9.4f}s {pct:>6.1f}% {bar}")
    
    print("\n" + "=" * 80)
    print("管道耗时估算（2000条messages）")
    print("=" * 80)
    
    # 估算管道总耗时
    # 假设: 600对话, 每对话2条可增强message, 3变体, 平均2.5步
    total_messages = 2000
    enhanceable_ratio = 0.4  # 约40%的message可增强
    enhanceable_msgs = int(total_messages * enhanceable_ratio)
    num_variants = 3
    avg_steps = 2.5  # min=2, max=3
    
    total_aug_calls = enhanceable_msgs * num_variants * avg_steps
    
    print(f"可增强messages: {enhanceable_msgs}")
    print(f"变体数: {num_variants}")
    print(f"平均步数: {avg_steps}")
    print(f"总增强调用次数: {total_aug_calls}")
    print()
    
    total_estimated_time = 0
    print(f"{'函数名':<25} {'调用次数':>10} {'总耗时':>10} {'占比':>8}")
    print("-" * 55)
    
    for name, r in sorted_results:
        calls = int(total_aug_calls / 9)  # 每个函数被调用的概率为1/9
        func_time = calls * r["avg"]
        total_estimated_time += func_time
    
    for name, r in sorted_results:
        calls = int(total_aug_calls / 9)
        func_time = calls * r["avg"]
        pct = func_time / total_estimated_time * 100 if total_estimated_time > 0 else 0
        bar = "█" * int(pct / 3)
        print(f"{name:<25} {calls:>10} {func_time:>9.1f}s {pct:>6.1f}% {bar}")
    
    print("-" * 55)
    print(f"{'总计':<25} {total_aug_calls:>10} {total_estimated_time:>9.1f}s")
    
    # 考虑重试的额外开销
    retry_multiplier = 1.5  # 估算重试带来的额外开销
    with_retries = total_estimated_time * retry_multiplier
    deepcopy_estimated = enhanceable_msgs * num_variants * 0.001  # 估算deepcopy
    
    print(f"\n考虑重试后估算: {with_retries:.0f}s ({with_retries/60:.1f}分钟)")
    print(f"deepcopy估算: {deepcopy_estimated:.0f}s")
    print(f"总估算: {with_retries + deepcopy_estimated:.0f}s ({(with_retries + deepcopy_estimated)/60:.1f}分钟)")
    
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    slowest = sorted_results[0]
    print(f"最慢的函数: {slowest[0]} (平均 {slowest[1]['avg']:.4f}s)")
    print(f"最慢函数占比: {slowest[1]['avg'] / total_avg * 100:.1f}%")
    
    if slowest[1]['avg'] / total_avg > 0.3:
        print(f"\n建议: 优先优化 {slowest[0]} 函数")
    
    asr_time = results.get("asr_noise", {}).get("avg", 0)
    asr_pct = asr_time / total_avg * 100
    print(f"\nasr_noise 占比: {asr_pct:.1f}%")
    if asr_pct < 30:
        print("asr_noise 不是主要瓶颈！其他增强函数才是。")
    else:
        print("asr_noise 是主要瓶颈之一。")


if __name__ == "__main__":
    main()
