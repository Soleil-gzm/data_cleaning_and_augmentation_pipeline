"""
文件工具模块
===========
提供文件统计与查找功能

注意：
  get_file_stats 在目录模式下只统计文件大小，不逐行数行数。
  逐行计数会在大量文件场景下导致严重的 I/O 抖动。
  如果确实需要行数统计，请使用 count_lines 单独统计。
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..exceptions import PipelineIOError


def count_lines(file_path: Union[str, Path]) -> int:
    """
    统计文件行数（快速）

    Args:
        file_path: 文件路径

    Returns:
        文件行数，文件不存在返回 0

    Raises:
        PipelineIOError: 读取失败
    """
    if not Path(file_path).exists():
        return 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except IOError as e:
        raise PipelineIOError(f"统计行数失败: {file_path}") from e


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    获取文件大小（MB）

    Args:
        file_path: 文件路径

    Returns:
        文件大小（MB），文件不存在返回 0.0
    """
    if not Path(file_path).exists():
        return 0.0
    return Path(file_path).stat().st_size / (1024 * 1024)


def get_file_stats(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    获取文件或目录的统计信息

    注意：目录模式下只统计文件大小，不逐行数行数。
    逐行计数会在大量文件场景下导致严重的 I/O 抖动。

    Args:
        file_path: 文件或目录路径

    Returns:
        统计信息字典：
        {
            "exists": bool,
            "size_mb": float,
            "is_dir": bool,
            "file_count": int（仅目录）,
        }
    """
    p = Path(file_path)
    if not p.exists():
        return {"exists": False, "size_mb": 0.0, "is_dir": False, "file_count": 0}
    if p.is_dir():
        total_size = 0.0
        file_count = 0
        for f in p.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size / (1024 * 1024)
                file_count += 1
        return {
            "exists": True,
            "size_mb": total_size,
            "is_dir": True,
            "file_count": file_count,
        }
    else:
        return {
            "exists": True,
            "size_mb": get_file_size_mb(p),
            "is_dir": False,
            "file_count": 1,
        }


def find_latest_file(
    directory: Union[str, Path], pattern: str = "*", sort_by_mtime: bool = True
) -> Optional[Path]:
    """
    在目录下查找符合模式的最新文件（按修改时间）

    Args:
        directory: 目录路径
        pattern: 文件匹配模式
        sort_by_mtime: 是否按修改时间排序

    Returns:
        最新文件的 Path，未找到返回 None
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
    "count_lines",
    "get_file_size_mb",
    "get_file_stats",
    "find_latest_file",
]