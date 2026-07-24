"""
写入模块：JSON/JSONL 写入相关
============================
包含所有写入操作，支持原子写入（临时文件 + os.rename）
"""

import json
import os
from pathlib import Path
from typing import Any, List, Dict, Optional, Union

from ..exceptions import PipelineIOError


def _atomic_write(file_path: Path, content: str, mode: str = "w"):
    """
    原子写入：先写入临时文件，再 rename 到目标文件
    防止写入过程中断导致文件损坏

    Args:
        file_path: 目标文件路径
        content: 要写入的内容
        mode: 写入模式

    Raises:
        PipelineIOError: 写入失败
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    try:
        with open(temp_path, mode, encoding="utf-8") as f:
            f.write(content)
        os.rename(temp_path, file_path)
    except IOError as e:
        if temp_path.exists():
            temp_path.unlink()
        raise PipelineIOError(f"写入文件失败: {file_path}") from e


def write_json(data: Any, file_path: Union[str, Path], indent: int = 2, atomic: bool = True):
    """
    写入 JSON 文件

    Args:
        data: JSON 数据
        file_path: 文件路径
        indent: 缩进空格数
        atomic: 是否使用原子写入（默认启用）

    Raises:
        PipelineIOError: 写入失败
    """
    file_path = Path(file_path)
    try:
        content = json.dumps(data, ensure_ascii=False, indent=indent)
        if atomic:
            _atomic_write(file_path, content)
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
    except IOError as e:
        raise PipelineIOError(f"写入文件失败: {file_path}") from e


def write_jsonl(data: List[Dict], file_path: Union[str, Path], atomic: bool = True):
    """
    写入 JSONL 文件

    Args:
        data: JSON 对象列表
        file_path: 文件路径
        atomic: 是否使用原子写入（默认启用）

    Raises:
        PipelineIOError: 写入失败
    """
    file_path = Path(file_path)
    try:
        content = "\n".join(json.dumps(item, ensure_ascii=False) for item in data) + "\n"
        if atomic:
            _atomic_write(file_path, content)
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
    except IOError as e:
        raise PipelineIOError(f"写入文件失败: {file_path}") from e


class JsonlWriter:
    """
    JSONL 流式写入器（内存友好）
    支持上下文管理器协议

    使用方式：
        with JsonlWriter("output.jsonl") as writer:
            writer.write(item1)
            writer.write(item2)
            writer.write_all([item3, item4, item5])

        # 或手动管理
        writer = JsonlWriter("output.jsonl")
        writer.write(item)
        writer.close()
    """

    def __init__(self, file_path: Union[str, Path], append: bool = False):
        self._file_path = Path(file_path)
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        try:
            self._f = open(self._file_path, mode, encoding="utf-8")
        except IOError as e:
            raise PipelineIOError(f"创建文件失败: {file_path}") from e
        self._count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write(self, item: Dict):
        """写入单条记录"""
        try:
            self._f.write(json.dumps(item, ensure_ascii=False) + "\n")
            self._count += 1
        except IOError as e:
            raise PipelineIOError(f"写入记录失败: {self._file_path}") from e

    def write_all(self, items: List[Dict]):
        """批量写入多条记录"""
        try:
            lines = [json.dumps(item, ensure_ascii=False) + "\n" for item in items]
            self._f.writelines(lines)
            self._count += len(items)
        except IOError as e:
            raise PipelineIOError(f"批量写入失败: {self._file_path}") from e

    def flush(self):
        """刷新缓冲区"""
        try:
            self._f.flush()
        except IOError as e:
            raise PipelineIOError(f"刷新缓冲区失败: {self._file_path}") from e

    def close(self):
        """关闭文件"""
        if hasattr(self, "_f") and self._f is not None:
            try:
                self._f.close()
            except IOError:
                pass
            self._f = None

    @property
    def count(self) -> int:
        """已写入记录数"""
        return self._count


def jsonl_writer(file_path: Union[str, Path], append: bool = False) -> JsonlWriter:
    """创建 JSONL 流式写入器"""
    return JsonlWriter(file_path, append)


__all__ = [
    "write_json",
    "write_jsonl",
    "JsonlWriter",
    "jsonl_writer",
]