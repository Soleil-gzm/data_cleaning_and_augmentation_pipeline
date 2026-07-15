#!/usr/bin/env python3
"""
完整流程性能分析：模拟2000条messages的实际处理
"""

import sys
import time
import json
import random
from pathlib import Path
from copy import deepcopy

scripts_dir = str(Path(__file__).parent)
sys.path.append(scripts_dir)

from common.asr_noise_augmenter import AsrNoiseAugmenter, print_asr_global_stats
from common import augment_utils_add as aug_utils


def create_sample_data(num_messages=2000):
    """创建模拟数据"""
    templates = [
        "我明天还款",
        "你好，请问有什么可以帮您",
        "这个分期方案怎么样",
        "逾期了怎么办",
        "最低还款额是多少",
        "华夏银行的信用卡",
        "协商一下还款计划",
        "利息太高了承受不了",
        "本金和利息分别是多少",
        "能否延期还款",
        "我现在手头比较紧",
        "下个月一定还上",
        "手续费能不能减免",
        "征信会受到影响吗",
        "个性化分期怎么办理",
    ]
    
    data = []
    for i in range(num_messages // 5):
        dialogue = {
            "id": i,
            "messages": []
        }
        for j in range(3):
            role = "user" if j % 2 == 0 else "assistant"
            content = random.choice(templates)
            dialogue["messages"].append({
                "role": role,
                "content": content,
                "loss": True
            })
        data.append(dialogue)
    return data


def setup_augmenter():
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
    return augmenter


def main():
    print("="*70)
    print("完整流程性能分析（模拟2000条messages）")
    print("="*70)
    
    num_messages = 2000
    print(f"准备模拟 {num_messages} 条messages...")
    
    data = create_sample_data(num_messages)
    print(f"创建了 {len(data)} 个对话，共 {sum(len(d['messages']) for d in data)} 条messages")
    
    setup_augmenter()
    print(f"增强器加载完成")
    
    config = {
        "target_roles": ["user"],
        "only_loss_true": True,
        "num_variants_per_dialogue": 3,
        "adaptive_variants": False,
        "message_augment_prob": 1.0,
        "augment_kwargs": {
            "num_variants": 1,
            "min_steps": 2,
            "max_steps": 3,
            "augment_weights": None
        }
    }
    
    rng = random.Random(42)
    
    print("\n" + "="*70)
    print("开始计时（只保留全局统计，关闭详细输出）")
    print("="*70)
    
    # 临时关闭详细计时输出
    import common.asr_noise_augmenter as asr_module
    original_print = asr_module.AsrNoiseAugmenter.find_best_abnormals
    
    def silent_find_best_abnormals(self, target_word, prev_word=None, top_k=5, alpha=0.7):
        _start = time.time()
        
        if prev_word and prev_word in self.prev_to_abnormals:
            candidates = self.prev_to_abnormals[prev_word]
        else:
            candidates = self.abnormal_words
        if not candidates:
            return []
        
        _encode_start = time.time()
        target_vec = self.encoder.encode([target_word])[0]
        _encode_time = time.time() - _encode_start
        
        scores = []
        _cos_start = time.time()
        for ab in candidates:
            idx = self.word_to_idx[ab]
            sem_sim = self._cosine_sim(target_vec, self.abnormal_vectors[idx])
            pin_sim = self._pinyin_similarity(target_word, ab)
            combined = alpha * pin_sim + (1 - alpha) * sem_sim
            scores.append((ab, combined))
        _cos_time = time.time() - _cos_start
        
        _sort_start = time.time()
        scores.sort(key=lambda x: x[1], reverse=True)
        _sort_time = time.time() - _sort_start
        
        _total_time = time.time() - _start
        
        asr_module._asr_global_stats["encode_calls"] += 1
        asr_module._asr_global_stats["encode_total_time"] += _encode_time
        asr_module._asr_global_stats["find_best_calls"] += 1
        asr_module._asr_global_stats["find_best_total_time"] += _total_time
        asr_module._asr_global_stats["similarity_total_time"] += _cos_time
        
        if asr_module._asr_global_stats["find_best_calls"] % 500 == 0:
            print(f"[PROGRESS] 已处理 {asr_module._asr_global_stats['find_best_calls']} 次 find_best_abnormals")
            print(f"           累计耗时: {asr_module._asr_global_stats['find_best_total_time']:.2f}s")
        
        return [ab for ab, _ in scores[:top_k]]
    
    asr_module.AsrNoiseAugmenter.find_best_abnormals = silent_find_best_abnormals
    
    total_start = time.time()
    
    all_original = []
    all_variants = []
    total_variants = 0
    
    step_times = {
        "deepcopy": 0.0,
        "augment_cell_multi": 0.0,
        "enhance_dialogue": 0.0,
        "other": 0.0
    }
    
    for idx, dialogue in enumerate(data):
        all_original.append(dialogue)
        
        try:
            enhance_start = time.time()
            
            messages = dialogue.get("messages", [])
            enhanceable = []
            for msg_idx, msg in enumerate(messages):
                if msg.get("role") in config["target_roles"]:
                    if msg.get("content", "").strip():
                        loss_val = msg.get("loss")
                        if isinstance(loss_val, str):
                            loss_val = loss_val.lower() == "true"
                        elif not isinstance(loss_val, bool):
                            loss_val = False
                        if loss_val:
                            enhanceable.append(msg_idx)
            
            if not enhanceable:
                continue
            
            num_variants = config["num_variants_per_dialogue"]
            aug_kwargs = config["augment_kwargs"]
            msg_prob = config.get("message_augment_prob", 1.0)
            
            for var_id in range(num_variants):
                dc_start = time.time()
                new_dialogue = deepcopy(dialogue)
                step_times["deepcopy"] += time.time() - dc_start
                
                new_messages = new_dialogue["messages"]
                selected = enhanceable[:]
                
                for msg_idx in selected:
                    if rng.random() > msg_prob:
                        continue
                    
                    original_text = new_messages[msg_idx].get("content", "")
                    if not original_text:
                        continue
                    
                    acm_start = time.time()
                    variants_list = aug_utils.augment_cell_multi(original_text, **aug_kwargs)
                    step_times["augment_cell_multi"] += time.time() - acm_start
                    
                    if variants_list and variants_list[0] != original_text:
                        new_messages[msg_idx]["content"] = variants_list[0]
                
                all_variants.append(new_dialogue)
                total_variants += 1
            
            step_times["enhance_dialogue"] += time.time() - enhance_start
            
        except Exception as e:
            print(f"对话 {idx} 增强失败: {e}")
            continue
        
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - total_start
            print(f"[PROGRESS] 已处理 {idx+1}/{len(data)} 个对话")
            print(f"           生成变体: {total_variants}")
            print(f"           耗时: {elapsed:.2f}s")
            print(f"           预计剩余: {elapsed/(idx+1)*(len(data)-idx-1):.2f}s")
    
    total_time = time.time() - total_start
    
    print("\n" + "="*70)
    print("性能分析结果")
    print("="*70)
    print(f"总对话数: {len(data)}")
    print(f"总messages数: {sum(len(d['messages']) for d in data)}")
    print(f"生成变体数: {total_variants}")
    print(f"总耗时: {total_time:.2f}s")
    print(f"平均每个对话耗时: {total_time/len(data):.4f}s")
    
    print("\n各步骤耗时:")
    print(f"  deepcopy: {step_times['deepcopy']:.2f}s ({step_times['deepcopy']/total_time*100:.1f}%)")
    print(f"  augment_cell_multi: {step_times['augment_cell_multi']:.2f}s ({step_times['augment_cell_multi']/total_time*100:.1f}%)")
    print(f"  enhance_dialogue: {step_times['enhance_dialogue']:.2f}s ({step_times['enhance_dialogue']/total_time*100:.1f}%)")
    
    print_asr_global_stats()
    
    print("\n" + "="*70)
    print("时间估算")
    print("="*70)
    stats = asr_module._asr_global_stats
    if stats["find_best_calls"] > 0:
        print(f"find_best_abnormals 调用次数: {stats['find_best_calls']}")
        print(f"平均每次耗时: {stats['find_best_total_time']/stats['find_best_calls']:.4f}s")
        print(f"\n如果只有asr_noise且无其他增强:")
        print(f"  纯asr_noise耗时: {stats['find_best_total_time']:.2f}s")
        print(f"  占总耗时比例: {stats['find_best_total_time']/total_time*100:.1f}%")
    
    print("\n" + "="*70)
    print("可能的瓶颈分析")
    print("="*70)
    if step_times["deepcopy"] / total_time > 0.2:
        print("  🔴 deepcopy 耗时占比较高，考虑减少变体数量或优化复制策略")
    if stats["find_best_total_time"] / total_time > 0.5:
        print("  🔴 asr_noise 是主要瓶颈，考虑添加向量缓存")
    if step_times["augment_cell_multi"] / total_time > 0.7:
        print("  🔴 multi_step_augment 调用其他增强操作也很耗时")
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()
