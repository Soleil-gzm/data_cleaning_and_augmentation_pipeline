import json
import re
from pathlib import Path

# ========== 硬编码配置 ==========
INPUT_FILE = "datas/suning_tools/data-record-processed-92049-filter.json"          # 你的输入文件
OUTPUT_FILE = "suning_tools/data-simulation-general2suning-260720_replaced.json"  # 输出路径
# ================================

# 两步正则：先处理“我的工号”+后续标点，再处理普通工号
PATTERN_MY = re.compile(r"[,，、。\s]*我的(?:的)?工号(?:是)?\s*\d+[,，、。]?")
PATTERN_OTHER = re.compile(r"[,，、。\s]*工号(?:是)?\s*\d+")

def should_keep_assistant_employee_id(messages, idx):
    """判断 idx 位置的 assistant 是否应保留工号（前一条 user 提到工号）"""
    if idx == 0:
        return False
    prev = messages[idx - 1]
    if prev.get("role") != "user":
        return False
    # 检测前一条 user 是否含有工号数字（不关心是否有“我的”）
    return bool(re.search(r"工号(?:是)?\s*\d+", prev["content"]))

def clean_text(text: str) -> str:
    """两步清理：先删“我的工号”+后续标点，再删普通工号（不删后续标点）"""
    # 第一步：删除“我的工号”模式（包括后面的逗号等）
    text = PATTERN_MY.sub("", text)
    # 第二步：删除其他工号模式（不删除后面标点）
    text = PATTERN_OTHER.sub("", text)
    return text

def process_dialogue(dialogue):
    messages = dialogue.get("messages", [])
    if not messages:
        return dialogue

    # 标记需要保留工号的 assistant
    keep_indices = set()
    for idx, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            if should_keep_assistant_employee_id(messages, idx):
                keep_indices.add(idx)

    # 执行删除
    for idx, msg in enumerate(messages):
        if msg.get("role") == "system":
            continue
        if idx in keep_indices:
            continue
        if "content" in msg:
            msg["content"] = clean_text(msg["content"])

    return dialogue

def process_json(data):
    if isinstance(data, list):
        return [process_dialogue(item) for item in data]
    elif isinstance(data, dict):
        return process_dialogue(data)
    else:
        raise ValueError("不支持的 JSON 格式")

def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"❌ 输入文件不存在: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_data = process_json(data)

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 处理完成！已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()