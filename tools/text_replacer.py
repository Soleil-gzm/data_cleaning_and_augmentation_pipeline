#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于正则的文本替换工具
使用方式：
    1. 修改脚本顶部的 RULES_FILE 和输入输出路径
    2. 运行 python text_replacer.py
"""

import json
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("❌ 缺少 pyyaml 依赖，请执行: pip install pyyaml")
    sys.exit(1)


# ========== 硬编码配置（直接修改这里） ==========
RULES_FILE = "tools/rules/notdue1.yaml"   # 规则文件路径
INPUT_FILE = "tools/datas/notdue/data-backbone-notdue-2w-cleaned-260723.json"            # 输入 JSON 文件路径
OUTPUT_FILE = "tools/result/data-backbone-notdue-2w-cleaned-260723_replace1.json"          # 输出 JSON 文件路径
# =============================================


class ReplacerRule:
    """单条替换规则"""
    def __init__(self, pattern: str, replacement: str, description: str = ''):
        self.pattern = re.compile(pattern)
        self.replacement = replacement
        self.description = description

    def apply(self, text: str) -> str:
        """如果匹配则替换，否则返回原文"""
        def repl(match):
            return self.replacement.format(**match.groupdict())
        new_text = self.pattern.sub(repl, text)
        return new_text


def load_rules(rules_file: Path) -> list:
    """加载 YAML 规则文件"""
    if not rules_file.exists():
        raise FileNotFoundError(f"规则文件不存在: {rules_file}")
    with open(rules_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    rules = []
    for r in data.get('rules', []):
        rules.append(ReplacerRule(
            pattern=r['pattern'],
            replacement=r['replacement'],
            description=r.get('description', '')
        ))
    return rules


def process_object(obj, rules):
    """递归处理 JSON 对象中的所有字符串"""
    if isinstance(obj, str):
        for rule in rules:
            new_text = rule.apply(obj)
            if new_text != obj:
                return new_text
        return obj
    elif isinstance(obj, list):
        return [process_object(item, rules) for item in obj]
    elif isinstance(obj, dict):
        return {k: process_object(v, rules) for k, v in obj.items()}
    else:
        return obj


def main():
    rules_file = Path(RULES_FILE)
    input_file = Path(INPUT_FILE)
    output_file = Path(OUTPUT_FILE)

    print(f"📌 规则文件: {rules_file}")
    print(f"📌 输入文件: {input_file}")
    print(f"📌 输出文件: {output_file}")

    # 加载规则
    try:
        rules = load_rules(rules_file)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    print(f"✅ 加载规则 {len(rules)} 条")

    # 读取输入 JSON
    if not input_file.exists():
        print(f"❌ 输入文件不存在: {input_file}")
        sys.exit(1)
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 处理
    processed = process_object(data, rules)

    # 写入输出
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"✅ 处理完成，已保存至: {output_file}")


if __name__ == "__main__":
    main()