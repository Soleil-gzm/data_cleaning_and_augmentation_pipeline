#!/usr/bin/env python3
"""
原始对话生成脚本（配置驱动版）
从 raw_dialogues_dir 读取 .doc/.docx 对话文件，
从 prompt_dir 读取对应的 system 信息 .txt 文件，
生成包含 loss="True" 的原始对话 JSON，输出到 output_file。
支持三种参数来源：
  1. --config_json (最高优先级，用于流水线集成)
  2. 独立命令行参数 (便于单独调试)
  3. 默认硬编码值 (向后兼容，会打印警告)
"""

import json
import os
import sys
import argparse
from pathlib import Path
from docx import Document

def extract_info(doc_path):
    """从 .doc 或 .docx 文件中提取对话轮次（保持不变）"""
    conversation = []
    if doc_path.endswith(".docx"):
        doc = Document(doc_path)
        for para in doc.paragraphs:
            text = para.text.strip()
            single_turn = {}
            if text.startswith("客户:"):
                single_turn["input"] = text.split(":", 1)[1].strip()
            elif text.startswith("客户："):
                single_turn["input"] = text.split("：", 1)[1].strip()
            elif text.startswith("专员:"):
                single_turn["output"] = text.split(":", 1)[1].strip()
            elif text.startswith("专员："):
                single_turn["output"] = text.split("：", 1)[1].strip()
            if single_turn:
                conversation.append(single_turn)
        if not conversation:
            print(f"警告: 文件 {doc_path} 中没有提取到任何对话内容")
            return None
        if list(conversation[0].keys())[0] == 'output':
            conversation.insert(0, {'input': ''})
        if list(conversation[-1].keys())[0] == 'input':
            conversation.pop()
    else:  # .doc 文件按纯文本读取
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        for text in lines:
            single_turn = {}
            if text.startswith("客户:"):
                single_turn["input"] = text.split(":", 1)[1].strip()
            elif text.startswith("客户："):
                single_turn["input"] = text.split("：", 1)[1].strip()
            elif text.startswith("专员:"):
                single_turn["output"] = text.split(":", 1)[1].strip()
            elif text.startswith("专员："):
                single_turn["output"] = text.split("：", 1)[1].strip()
            if single_turn:
                conversation.append(single_turn)
        if not conversation:
            print(f"警告: 文件 {doc_path} 中没有提取到任何对话内容")
            return None
        if list(conversation[0].keys())[0] == 'output':
            conversation.insert(0, {'input': ''})
        if list(conversation[-1].keys())[0] == 'input':
            conversation.pop()

    # 检查对话顺序是否正确
    for i in range(len(conversation)):
        current_keys = list(conversation[i].keys())
        if i % 2 == 0:  # 应该是 input
            if 'input' not in current_keys:
                print(f"警告: 文件 {doc_path} 对话顺序错误，期待input但未找到")
                return None
        else:  # 应该是 output
            if 'output' not in current_keys:
                print(f"警告: 文件 {doc_path} 对话顺序错误，期待output但未找到")
                return None

    if len(conversation) % 2 != 0:
        print(f"警告: 文件 {doc_path} 对话轮次为奇数，已自动忽略最后一轮（不完整）")
        conversation = conversation[:-1]  # 去掉最后一个不完整的input

    # 将连续两轮（input+output）合并为一个字典
    dialogs = [{**conversation[idx], **conversation[idx+1]} for idx in range(0, len(conversation), 2)]
    return dialogs

def reformat_dialogs(dialogs):
    """将 dialogs 列表转换为标准的 messages 格式，为每个 assistant 添加 loss="True" """
    messages = []
    if "system" in dialogs[0]:
        messages.append({"role": "system", "content": dialogs[0]["system"]})

    for item in dialogs:
        if "input" in item:
            messages.append({"role": "user", "content": item["input"]})
        if "output" in item:
            messages.append({"role": "assistant", "content": item["output"], "loss": "True"})

    return {"messages": messages}

def main():
    parser = argparse.ArgumentParser(description="生成原始对话JSON")
    parser.add_argument("--config_json", type=str, help="全局配置JSON字符串（优先级最高）")
    parser.add_argument("--raw_dir", type=str, help="存放 .doc/.docx 文件的目录")
    parser.add_argument("--prompt_dir", type=str, help="存放 case_*.txt 文件的目录")
    parser.add_argument("--output_file", type=str, help="输出的 JSON 文件路径")
    args = parser.parse_args()

    # ---------- 参数解析优先级 ----------
    if args.config_json:
        # 从配置JSON中读取路径
        config = json.loads(args.config_json)
        task_name = config['task_name']
        base_dir = Path(config['paths']['output']['base_dir'])
        # 注意：步骤00的输入输出路径可能定义在 config['steps']['00_generate_raw'] 中
        step_cfg = config.get('steps', {}).get('00_generate_raw', {})
        raw_dir = step_cfg.get('raw_dir') or config['paths']['input']['raw_dialogues_dir']
        prompt_dir = step_cfg.get('prompt_dir') or config['paths']['input']['prompt_dir']
        output_file = step_cfg.get('output_file') or str(base_dir / task_name / "raw_dialogues.json")
    else:
        # 使用独立命令行参数
        raw_dir = args.raw_dir
        prompt_dir = args.prompt_dir
        output_file = args.output_file
        # 若均未提供，则退化为硬编码默认值（并警告）
        if not raw_dir or not prompt_dir or not output_file:
            print("警告: 未提供 --config_json 且缺少 --raw_dir/--prompt_dir/--output_file，将使用硬编码默认值")
            raw_dir = raw_dir or "./data/Yangqg_simulation_data"
            prompt_dir = prompt_dir or "./data/cases_random"
            output_file = output_file or "./intermediate/raw_dialogues.json"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"使用配置: raw_dir={raw_dir}, prompt_dir={prompt_dir}, output={output_file}")

    multi_dialogs = []
    for filename in os.listdir(raw_dir):
        if not (filename.endswith(".docx") or filename.endswith(".doc")):
            continue
        file_path = os.path.join(raw_dir, filename)
        dialogs = extract_info(doc_path=file_path)
        if dialogs is None:
            print(f"Skip: {filename}")
            continue

        # 从文件名中提取案例编号（假设格式为 "案例<数字>.docx"）
        try:
            # 兼容 "案例123.docx" 或 "case_123.docx" 等多种格式
            if "案例" in filename:
                simulation_id = int(filename.split("案例")[1].split(".")[0])
            elif "case_" in filename:
                simulation_id = int(filename.split("case_")[1].split(".")[0])
            else:
                print(f"无法从文件名解析案例ID: {filename}，跳过")
                continue
        except Exception as e:
            print(f"文件名解析失败: {filename}, 错误: {e}")
            continue
        case_id = simulation_id

        prompt_file = os.path.join(prompt_dir, f"case_{case_id}.txt")
        if not os.path.exists(prompt_file):
            print(f"警告: system 文件不存在 {prompt_file}，跳过 {filename}")
            continue
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read()

        # 构建 system 消息（使用读取的 prompt 内容）
        system_content = (
            "# 角色\n"
            "    你是一位来自华夏银行委外阳光机构的电话专员，现在你需要向以下客户催收信用卡欠款。\n\n"
            "## 案例信息\n\n"
            "## 标签信息\n\n"
            + prompt      # 去掉prompt
        )
        dialogs[0]["system"] = system_content

        # 调整键顺序（使 system 在前）
        dialogs[0] = {key: dialogs[0][key] for key in ["system", "input", "output"]}

        messages = reformat_dialogs(dialogs)
        multi_dialogs.append(messages)

    with open(output_file, "wt", encoding="utf-8") as file:
        json.dump(multi_dialogs, file, ensure_ascii=False, indent=4)

    print(f"原始对话数据已生成: {output_file} (共 {len(multi_dialogs)} 个对话)")

if __name__ == "__main__":
    main()