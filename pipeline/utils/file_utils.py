"""
文件操作工具：JSON/JSONL 读写、统计、查找最新文件
"""

import json
from pathlib import Path
from typing import Any, List, Dict, Optional, Union


def read_json(file_path: Union[str, Path]) -> Any:
    """读取 JSON 文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, file_path: Union[str, Path], indent: int = 2):
    """写入 JSON 文件"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    """读取 JSONL 文件，返回列表"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(data: List[Dict], file_path: Union[str, Path]):
    """写入 JSONL 文件"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def count_lines(file_path: Union[str, Path]) -> int:
    """统计文件行数（快速）"""
    if not Path(file_path).exists():
        return 0
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """获取文件大小（MB）"""
    if not Path(file_path).exists():
        return 0.0
    return Path(file_path).stat().st_size / (1024 * 1024)


def get_file_stats(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    获取文件或目录的统计信息
    返回: {
        "exists": bool,
        "lines": int (如果是目录则递归统计所有文件行数),
        "size_mb": float,
        "is_dir": bool
    }
    """
    p = Path(file_path)
    if not p.exists():
        return {"exists": False, "lines": 0, "size_mb": 0.0, "is_dir": False}
    if p.is_dir():
        total_lines = 0
        total_size = 0.0
        for f in p.rglob("*"):
            if f.is_file():
                total_lines += count_lines(f)
                total_size += get_file_size_mb(f)
        return {
            "exists": True,
            "lines": total_lines,
            "size_mb": total_size,
            "is_dir": True,
        }
    else:
        return {
            "exists": True,
            "lines": count_lines(p),
            "size_mb": get_file_size_mb(p),
            "is_dir": False,
        }


def find_latest_file(
    directory: Union[str, Path], pattern: str = "*", sort_by_mtime: bool = True
) -> Optional[Path]:
    """
    在目录下查找符合模式的最新文件（按修改时间）
    返回 Path 或 None
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return None
    files = list(dir_path.glob(pattern))
    if not files:
        return None
    if sort_by_mtime:
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


__all__ = [
    "read_json",
    "write_json",
    "read_jsonl",
    "write_jsonl",
    "count_lines",
    "get_file_size_mb",
    "get_file_stats",
    "find_latest_file",
]