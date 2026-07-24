"""
统一 IO 模块
===========
提供统一的文件读写接口，支持：
- JSON/JSONL 批量读写
- JSONL 流式读写（支持大文件处理）
- 文件统计与查找
- pickle 序列化

使用方式：
    from pipeline.io import read_json, write_json, read_jsonl, write_jsonl
    from pipeline.io import jsonl_reader, jsonl_writer, JsonlWriter

    # 批量读写
    data = read_json("data.json")
    write_json(data, "output.json")

    # 流式读取（内存友好）
    for item in jsonl_reader("large_data.jsonl"):
        process(item)

    # 流式写入（内存友好）
    with JsonlWriter("output.jsonl") as writer:
        for item in items:
            writer.write(item)
"""

# JSON/JSONL 读取
from .reader import read_json, read_jsonl, jsonl_reader, jsonl_chunker

# JSON/JSONL 写入
from .writer import write_json, write_jsonl, JsonlWriter, jsonl_writer

# Pickle 序列化
from .pickle_io import read_pickle, write_pickle

# 文件统计与查找
from .file_utils import count_lines, get_file_size_mb, get_file_stats, find_latest_file

__all__ = [
    # JSON 批量读写
    "read_json",
    "write_json",
    # JSONL 批量读写
    "read_jsonl",
    "write_jsonl",
    # JSONL 流式读取
    "jsonl_reader",
    "jsonl_chunker",
    # JSONL 流式写入
    "JsonlWriter",
    "jsonl_writer",
    # Pickle 序列化
    "read_pickle",
    "write_pickle",
    # 文件统计与查找
    "count_lines",
    "get_file_size_mb",
    "get_file_stats",
    "find_latest_file",
]