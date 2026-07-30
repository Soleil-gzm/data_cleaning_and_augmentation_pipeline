import json
import re
from pathlib import Path

# ========== 硬编码配置 ==========
INPUT_FILE = "datas/suning_tools/data-record-processed-92049-filter.json"          # 你的输入文件
OUTPUT_FILE = "suning_tools/data-record-processed-92049-filter_replaced.json"  # 输出路径
# ================================

# 1. 匹配“我的工号”格式，包括后面的逗号等（需删除后续标点）
PATTERN_MY = re.compile(r"[,，、。\s]*我的(?:的)?工号(?:是)?\s*\d+[,，、。]?")

# 2. 匹配普通工号格式（不含后续标点，保留标点）
PATTERN_OTHER = re.compile(r"[,，、。\s]*工号(?:是)?\s*\d+")

# 3. 匹配尾号格式（包括前面的标点和后面的空格或“的”）
PATTERN_TAIL = re.compile(r"[,，、。\s]*尾号\s*\d+\s*(?:的)?\s*")

def should_keep_assistant_employee_id(messages, idx):
    """判断 idx 位置的 assistant 是否应保留工号（前一条 user 提到工号）"""
    if idx == 0:
        return False
    prev = messages[idx - 1]
    if prev.get("role") != "user":
        return False
    # 只要 user 提到“工号”就触发保留
    return "工号" in prev["content"]

def clean_text(text: str) -> str:
    """三步清理：先删“我的工号”+后续标点，再删普通工号，最后删尾号"""
    text = PATTERN_MY.sub("", text)
    text = PATTERN_OTHER.sub("", text)
    text = PATTERN_TAIL.sub("", text)
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