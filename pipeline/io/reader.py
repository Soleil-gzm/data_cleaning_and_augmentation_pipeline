"""
读取模块：JSON/JSONL 读取相关
============================
包含所有读取操作，统一异常处理
"""

import json
from pathlib import Path
from typing import Any, List, Dict, Optional, Union, Iterator, Generator

from ..exceptions import PipelineIOError


def read_json(file_path: Union[str, Path]) -> Any:
    """
    读取 JSON 文件

    Args:
        file_path: 文件路径

    Returns:
        JSON 数据

    Raises:
        PipelineIOError: 文件不存在或 JSON 解析失败
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise PipelineIOError(f"文件不存在: {file_path}") from e
    except json.JSONDecodeError as e:
        raise PipelineIOError(f"JSON 解析失败: {file_path}") from e
    except IOError as e:
        raise PipelineIOError(f"读取文件失败: {file_path}") from e


def read_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    """
    读取 JSONL 文件，返回列表

    Args:
        file_path: 文件路径

    Returns:
        JSON 对象列表

    Raises:
        PipelineIOError: 文件不存在或解析失败
    """
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise PipelineIOError(f"JSONL 行解析失败: {file_path}") from e
        return data
    except FileNotFoundError as e:
        raise PipelineIOError(f"文件不存在: {file_path}") from e
    except IOError as e:
        raise PipelineIOError(f"读取文件失败: {file_path}") from e


def jsonl_reader(file_path: Union[str, Path]) -> Iterator[Dict]:
    """
    流式读取 JSONL 文件（内存友好）
    返回迭代器，逐行读取

    Args:
        file_path: 文件路径

    Yields:
        单个 JSON 对象

    Raises:
        PipelineIOError: 文件不存在或解析失败
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        raise PipelineIOError(f"JSONL 行解析失败: {file_path}") from e
    except FileNotFoundError as e:
        raise PipelineIOError(f"文件不存在: {file_path}") from e
    except IOError as e:
        raise PipelineIOError(f"读取文件失败: {file_path}") from e


def jsonl_chunker(
    file_path: Union[str, Path], chunk_size: int = 1000
) -> Generator[List[Dict], None, None]:
    """
    分块读取 JSONL 文件（内存友好）
    每 chunk_size 条记录返回一次

    Args:
        file_path: 文件路径
        chunk_size: 每块的记录数

    Yields:
        JSON 对象列表（每块）

    Raises:
        PipelineIOError: 文件不存在或解析失败
    """
    chunk = []
    for item in jsonl_reader(file_path):
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


__all__ = [
    "read_json",
    "read_jsonl",
    "jsonl_reader",
    "jsonl_chunker",
]