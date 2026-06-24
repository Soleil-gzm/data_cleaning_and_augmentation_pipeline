"""
配置加载器：支持分层新结构，同时向后兼容旧扁平结构
"""
import yaml
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


class ConfigLoader:
    @staticmethod
    def load(config_path: Path) -> Dict[str, Any]:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # 变量替换
        task_name = raw.get("task_name", "default_task")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        def recursive_replace(obj):
            if isinstance(obj, dict):
                return {k: recursive_replace(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_replace(item) for item in obj]
            elif isinstance(obj, str):
                try:
                    return obj.format(task_name=task_name, timestamp=timestamp)
                except KeyError:
                    return obj
            else:
                return obj

        config = recursive_replace(raw)

        # 补全新结构默认值（若缺失）
        config = ConfigLoader._ensure_new_structure(config)

        return config

    @staticmethod
    def _ensure_new_structure(config: Dict) -> Dict:
        """自动补全新结构字段，兼容旧配置"""
        # 如果 paths 不存在，从旧结构推导
        if "paths" not in config:
            base = config.get("paths", {}).get("output", {}).get("base_dir", "./intermediate")
            config["paths"] = {
                "input": {
                    "raw_dialogues_dir": config.get("raw_dir", "./data/Yangqg_simulation_data"),
                    "prompt_dir": config.get("prompt_dir", "./data/cases_random"),
                },
                "intermediate": base,
                "output": "./output"
            }

        # 如果 reporting 不存在，补默认
        if "reporting" not in config:
            config["reporting"] = {
                "enabled": True,
                "reporters": [
                    {"type": "json", "output_dir": "{task_dir}/reports/"},
                    {"type": "csv", "output_dir": "{task_dir}/reports/"},
                    {"type": "matplotlib", "output_dir": "{task_dir}/reports/plots/", "dpi": 150}
                ]
            }

        # 如果 executor 不存在
        if "executor" not in config:
            config["executor"] = {"type": "sequential", "max_workers": 4}

        # 如果 logging 不存在
        if "logging" not in config:
            config["logging"] = {"level": "INFO", "file_level": "DEBUG", "show_progress": True, "print_tree": True}

        # 保证 steps 存在
        if "steps" not in config:
            config["steps"] = {}

        # 为每个步骤补默认 enabled=True
        for step_name in config["steps"]:
            if "enabled" not in config["steps"][step_name]:
                config["steps"][step_name]["enabled"] = True

        return config