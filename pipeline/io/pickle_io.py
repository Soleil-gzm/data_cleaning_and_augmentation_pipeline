"""
Pickle 序列化模块
=================
提供 pickle 序列化支持

安全警告：
  pickle 反序列化可以执行任意代码。
  仅用于加载内部生成的可信数据，禁止加载外部来源的 pickle 文件。
  对于生产环境，更推荐用 JSON 或 Parquet 替代 Pickle。
"""

import pickle
from pathlib import Path
from typing import Any, Union

from ..exceptions import PipelineIOError


def read_pickle(file_path: Union[str, Path]) -> Any:
    """
    读取 pickle 文件

    安全警告：pickle 反序列化可以执行任意代码。
    仅用于加载内部生成的可信数据，禁止加载外部来源的 pickle 文件。

    Args:
        file_path: 文件路径

    Returns:
        反序列化的数据

    Raises:
        PipelineIOError: 文件不存在或反序列化失败
    """
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError as e:
        raise PipelineIOError(f"文件不存在: {file_path}") from e
    except pickle.UnpicklingError as e:
        raise PipelineIOError(f"Pickle 反序列化失败: {file_path}") from e
    except IOError as e:
        raise PipelineIOError(f"读取文件失败: {file_path}") from e


def write_pickle(data: Any, file_path: Union[str, Path], protocol: int = pickle.HIGHEST_PROTOCOL):
    """
    写入 pickle 文件

    Args:
        data: 要序列化的数据
        file_path: 文件路径
        protocol: pickle 协议版本

    Raises:
        PipelineIOError: 写入失败
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(file_path, "wb") as f:
            pickle.dump(data, f, protocol=protocol)
    except IOError as e:
        raise PipelineIOError(f"写入文件失败: {file_path}") from e


__all__ = [
    "read_pickle",
    "write_pickle",
]