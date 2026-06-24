"""
文件操作工具：JSON/JSONL 读写、统计、目录树打印
"""
import json
import os
import fnmatch
from pathlib import Path
from typing import Any, List, Dict, Optional, Union
import shutil


def read_json(file_path: Union[str, Path]) -> Any:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, file_path: Union[str, Path], indent: int = 2):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(data: List[Dict], file_path: Union[str, Path]):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def count_lines(file_path: Union[str, Path]) -> int:
    if not Path(file_path).exists():
        return 0
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    if not Path(file_path).exists():
        return 0.0
    return Path(file_path).stat().st_size / (1024 * 1024)


def get_file_stats(file_path: Union[str, Path]) -> Dict[str, Any]:
    """获取文件统计：行数、大小MB、是否存在"""
    p = Path(file_path)
    if not p.exists():
        return {"exists": False, "lines": 0, "size_mb": 0.0}
    if p.is_dir():
        total_lines = 0
        total_size = 0.0
        for f in p.rglob("*"):
            if f.is_file():
                total_lines += count_lines(f)
                total_size += get_file_size_mb(f)
        return {"exists": True, "lines": total_lines, "size_mb": total_size, "is_dir": True}
    else:
        return {"exists": True, "lines": count_lines(p), "size_mb": get_file_size_mb(p), "is_dir": False}


def print_directory_tree(
    path: Union[str, Path],
    max_depth: int = 3,
    prefix: str = "",
    exclude_patterns: List[str] = None,
    show_files: bool = True,
    _depth: int = 0
):
    """
    打印目录结构树
    """
    if exclude_patterns is None:
        exclude_patterns = [".step_*", "__pycache__", "*.pyc", ".DS_Store", "*.log"]

    def _is_excluded(name: str) -> bool:
        for pat in exclude_patterns:
            if fnmatch.fnmatch(name, pat):
                return True
        return False

    p = Path(path)
    if not p.exists():
        print(f"{prefix}❌ 目录不存在: {p}")
        return

    if _depth == 0:
        print(f"📁 {p.name}/")

    if _depth >= max_depth:
        if any(p.iterdir()):
            print(f"{prefix}  └── ... (深度限制 {max_depth})")
        return

    items = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
    items = [item for item in items if not _is_excluded(item.name)]

    for idx, item in enumerate(items):
        is_last = (idx == len(items) - 1)
        connector = "└── " if is_last else "├── "
        line = f"{prefix}{connector}"

        if item.is_dir():
            print(f"{line}📁 {item.name}/")
            extension = "    " if is_last else "│   "
            print_directory_tree(item, max_depth, prefix + extension, exclude_patterns, show_files, _depth + 1)
        else:
            if show_files:
                size = get_file_size_mb(item)
                print(f"{line}📄 {item.name} ({size:.2f} MB)")