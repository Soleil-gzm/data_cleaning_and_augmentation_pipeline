#!/usr/bin/env python3
"""
原始对话生成脚本
从 Yangqg_simulation_data 目录读取 .doc/.docx 对话文件，
从 cases_random 目录读取对应的 system 信息 .txt 文件，
生成包含 loss="True" 的原始对话 JSON，输出到 intermediate/raw_dialogues.json
"""

import json
import os
from docx import Document

def extract_info(doc_path):
    """
    从 .doc 或 .docx 文件中提取对话轮次
    返回: list of dict, 每个 dict 包含 "input" 和 "output" 键
    """
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
            conversation.pop()  # 删除最后一个元素，即客户说的话---需要保证是废话
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
        print(f"警告: 文件 {doc_path} 对话轮次为奇数，添加空的output")
        return None

    # 将连续两轮（input+output）合并为一个字典
    dialogs = [{**conversation[idx], **conversation[idx+1]} for idx in range(0, len(conversation), 2)]
    return dialogs

def reformat_dialogs(dialogs):
    """
    将 dialogs 列表转换为标准的 messages 格式
    为每个 assistant 消息添加 loss="True"
    """
    messages = []
    if "system" in dialogs[0]:
        messages.append({"role": "system", "content": dialogs[0]["system"]})

    for item in dialogs:
        if "input" in item:
            messages.append({"role": "user", "content": item["input"]})
        if "output" in item:
            messages.append({"role": "assistant", "content": item["output"], "loss": "True"})

    return {"messages": messages}

if __name__ == "__main__":
    # 原始数据目录（根据你的实际存放位置）
    directory = "./data/Yangqg_simulation_data"
    prompt_dir = "./data/cases_random"
    output_dir = "./intermediate"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "raw_dialogues.json")

    multi_dialogs = []
    for filename in os.listdir(directory):
        if filename.endswith(".docx") or filename.endswith(".doc"):
            file_path = os.path.join(directory, filename)
            dialogs = extract_info(doc_path=file_path)
            if dialogs is None:
                print(f"Skip: {filename}")
                continue

            # 从文件名中提取案例编号（假设格式为 "案例<数字>.docx"）
            try:
                simulation_id = int(filename.split("案例")[1].split(".")[0])
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

            # 注入 system 信息
            # dialogs[0]["system"] = "# 角色\n    你是一位来自华夏银行委外阳光机构的电话专员，现在你需要向以下客户催收信用卡欠款。\n\n## 案例信息\n\n## 标签信息\n\n" + prompt
            
            dialogs[0]["system"] = "# 角色\n    你是一位来自华夏银行委外阳光机构的电话专员，现在你需要向以下客户催收信用卡欠款。\n\n## 案例信息\n\n## 标签信息\n\n" 

            # 调整键顺序（使 system 在前）
            dialogs[0] = {key: dialogs[0][key] for key in ["system", "input", "output"]}

            messages = reformat_dialogs(dialogs)
            multi_dialogs.append(messages)

    with open(output_file, "wt", encoding="utf-8") as file:
        json.dump(multi_dialogs, file, ensure_ascii=False, indent=4)

    print(f"原始对话数据已生成: {output_file} (共 {len(multi_dialogs)} 个对话)")