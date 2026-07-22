"""
进度条工具：基于 tqdm，支持全局开关
"""

import sys
from typing import Iterable, Optional
from tqdm import tqdm as _tqdm

_SHOW_PROGRESS = True


def set_progress_global(show: bool):
    global _SHOW_PROGRESS
    _SHOW_PROGRESS = show


def get_progress_bar(
    iterable: Iterable,
    desc: str = "Processing",
    unit: str = "it",
    total: Optional[int] = None,
    show: Optional[bool] = None,
    **kwargs
):
    """
    获取进度条迭代器
    """
    if show is None:
        show = _SHOW_PROGRESS

    if not show:
        return iterable

    return _tqdm(iterable, desc=desc, unit=unit, total=total, file=sys.stdout, **kwargs)