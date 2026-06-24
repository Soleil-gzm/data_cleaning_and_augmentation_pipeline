"""
配置加载器：支持 {task_name}、{timestamp}、{task_dir} 占位符
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


class ConfigLoader:
    @staticmethod
    def load(config_path: Path) -> Dict[str, Any]:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        task_name = raw.get("task_name", "default_task")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 计算 task_dir（如果配置中有 paths.intermediate）
        intermediate = raw.get("paths", {}).get("intermediate", "./intermediate")
        task_dir = Path(intermediate) / task_name

        def recursive_replace(obj):
            if isinstance(obj, dict):
                return {k: recursive_replace(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_replace(item) for item in obj]
            elif isinstance(obj, str):
                # 同时替换 {task_dir}，确保路径正确
                return obj.format(
                    task_name=task_name, timestamp=timestamp, task_dir=str(task_dir)
                )
            else:
                return obj

        config = recursive_replace(raw)

        # 补全默认结构（可选）
        config = ConfigLoader._ensure_defaults(config)

        return config

    @staticmethod
    def _ensure_defaults(config: Dict) -> Dict:
        """补全缺失字段，确保兼容"""
        if "paths" not in config:
            config["paths"] = {
                "intermediate": "./intermediate",
                "input": {},
                "output": "./output",
            }
        if "steps" not in config:
            config["steps"] = {}
        if "logging" not in config:
            config["logging"] = {
                "level": "INFO",
                "file_level": "DEBUG",
                "show_progress": True,
            }
        return config
